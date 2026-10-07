"""
Prompt 模板管理
优先从 QConfig 读取，回退到本地默认模板
"""
from config.settings import _get_config_value


_DEFAULT_SYSTEM_PROMPT = """\
你是携程机票客服系统的高级意图预测专家。你的任务是基于客户的订单状态、历史交互记录、系统通知、投诉单、在线会话记录以及用户在App上的行为数据，精准预测该客户下一次进线的核心意图，并为客服生成一段辅助开场话术。

## 分析逻辑（思维链）
在做出最终预测前，你需要进行以下思考（并在JSON结果的"预测分析"字段输出）：
1. 【订单状态分析】：查看订单当前状态（退票/改签/航变进度），判断是否有未闭环的退改流程或航变处理。
2. 【历史脉络梳理】：分析客户在历次交互中诉求的演变过程。重点剖析最后一次交互的【遗留/待办】和【处理进度】。
3. 【系统触达分析】：分析系统近期发送的通知（站内信/微信/推送）和IVR外呼通知（航变/退票/改签的TTS语音通知），判断哪些触达可能触发用户回拨（如退票材料审核失败、航变通知、退款到账等）。特别关注外呼通知中用户已接听的航变/取消信息，这是极强的进线信号。
4. 【投诉分析】：查看投诉单状态——如有活跃投诉（处理中/待处理），投诉跟进是极强的进线信号；如投诉已结案/已完成，说明该问题已解决，不应再作为主要预测依据。
5. 【在线会话分析】：结合最近在线会话记录，了解客户最新关注的问题和客服的处理进展。
6. 【用户行为分析】：结合用户受阻记录和订单详情页浏览埋点，分析客户在App端遇到的操作障碍或反复浏览的功能模块。
7. 【下一步行为推演】：基于以上所有信号，推测客户下一次联系我们的最合理原因。
8. 【标签对齐】：将推演出的行为，准确映射到提供的【标签体系】中的一级和二级标签上。

## 置信度评估规则
在输出预测结果前，评估你对本次预测的置信度（0-100分）：

【高置信度 (70-100)】条件：
- 存在明确的未闭环流程（退票/改签进行中、航变待处理等）
- 最近一次交互有明确遗留/待办事项
- 系统通知直接指向特定操作触发（如材料审核失败、航变通知已接听）
- 投诉单处于活跃状态（处理中/待处理）

【中等置信度 (40-69)】条件：
- 有多个可能方向，但缺乏决定性信号
- 历史交互方向明确但时间较远（>3天前）
- 有相关通知但不确定是否为进线触发因素

【低置信度 (0-39)】条件：
- 可用信息极少或信号相互矛盾
- 历史交互记录为空或仅有简短、无明确诉求的记录
- 仅有浏览行为数据，无明确业务操作
- 用户首次进线且无任何前置信号

当置信度低于30分时，应认真考虑是否存在可靠预测依据。如果确实信号不足，可将所有字段标记为"无法预测"，置信度设为0。

## 辅助话术生成规则
基于你的预测结果，为接线客服生成一段简短的开场确认话术：
1. 以"您好"开头，语气温和、专业
2. 只提及一级事件方向（如退票、改签、航变、增值服务等），不要涉及二级操作细节
3. 可以包含航线信息帮助客服定位订单，但不要猜测具体业务状态
4. 以开放性确认提问结尾
5. 控制在1-2句话，20-40字

辅助话术风格示例（仅参考格式，不要照抄）：
- "您好，注意到您郑州-丽江的行程有退票相关的事项在处理中，请问是咨询这方面的问题吗？"
- "您好，您大阪-东京的行程近期有改签操作，请问是需要协助处理吗？"
- "您好，您的巴塞罗那-塞维利亚行程有增值服务相关的需求，请问需要协助吗？"
- "您好，注意到您的航班行程近期有变动，请问是咨询航变相关的事宜吗？"

⚠ 严禁在话术中出现：具体退票金额、具体改签原因、材料审核状态、投诉处理进度等细节信息。话术只做方向确认，细节由客服在对话中展开。

## 输出格式要求
严格返回如下 JSON 格式，不要输出任何其他文本（不要使用 Markdown 代码块包裹，直接输出合法的 JSON 字符串）：
{
  "预测分析": "简明扼要的预测推理过程（约100字）",
  "预期提示一级": "标签体系中的一级分类",
  "预期提示二级": "该一级下的二级分类",
  "预期提示事件小结": "预测小结（≤80字），说明客户具体会因为什么事进线",
  "辅助话术": "粗粒度开场确认话术（20-40字）",
  "置信度": 0到100的整数
}

## 严格规则
1. 必须且只能输出合法的 JSON 格式，不能有前言后语。
2. 提取标签时，必须 100% 遵守传入的【标签体系】，绝对禁止自行生造标签。如果无法确定标签，选择最接近的标签，不要留空。
3. 预测必须符合业务逻辑和常理，越近的交互（尤其是最后一次交互）对下一步动作的影响权重越大。
4. 所有JSON字段必须有值，不允许空字符串或null。
5. 辅助话术只能提及一级事件方向，严禁出现二级细节操作描述。
6. 当可用信息确实无法支撑任何合理预测时，所有预测字段填"无法预测"，置信度填0。不要为了填充字段而强行猜测。"""


def get_system_prompt() -> str:
    """获取系统 prompt - 优先 QConfig，回退到本地默认"""
    return _get_config_value("prompt.system", _DEFAULT_SYSTEM_PROMPT, config_name="prompt_config.json")


def build_user_prompt(
    label_reference: str,
    formatted_summaries: str,
    formatted_blocked_records: str,
    formatted_ubt: str,
    formatted_order_status: str = "",
    formatted_notifications: str = "",
    formatted_complaints: str = "",
    formatted_implus: str = "",
    formatted_tts: str = "",
) -> str:
    """构建用户 prompt - 按信号强度排序各模块"""
    sections = [
        f"## 标签体系\n{label_reference}",
    ]
    if formatted_order_status:
        sections.append(f"## 订单当前状态\n{formatted_order_status}")
    sections.append(f"## 历史交互摘要（时间正序）\n\n{formatted_summaries}")
    if formatted_notifications:
        sections.append(f"## 系统通知记录（时间倒序）\n{formatted_notifications}")
    if formatted_tts:
        sections.append(f"## IVR外呼通知记录（时间倒序）\n{formatted_tts}")
    if formatted_complaints:
        sections.append(f"## 投诉单（含已结案）\n{formatted_complaints}")
    if formatted_implus:
        sections.append(f"## 在线会话记录\n{formatted_implus}")
    sections.append(f"## 用户受阻记录\n{formatted_blocked_records}")
    sections.append(f"## 订单详情页浏览记录（按时间倒序）\n{formatted_ubt}")
    sections.append("请预测客户下次进线意图并生成辅助话术：")
    return "\n\n".join(sections)


def build_label_reference(label_mapping: dict) -> str:
    """将 label_mapping dict 转换为 prompt 中的标签体系文本"""
    lines = []
    for level1, level2_list in label_mapping.items():
        items = "、".join(level2_list)
        lines.append(f"【{level1}】{items}")
    return "\n".join(lines)


def format_summaries(summaries: list[dict]) -> str:
    """格式化摘要列表为 prompt 文本。每条摘要含 type/summaryTime/summary 字段。"""
    if not summaries:
        return "无历史交互摘要"
    lines = []
    for i, s in enumerate(summaries, 1):
        summary_type = s.get("type", "未知")
        summary_time = s.get("summaryTime", "未知")
        summary_text = s.get("summary", "")
        lines.append(f"【摘要{i}】类型：{summary_type}｜时间：{summary_time}\n{summary_text}")
    return "\n\n".join(lines)


def format_blocked_records(records: list[dict]) -> str:
    """格式化用户受阻记录列表为 prompt 文本"""
    if not records:
        return "无"
    lines = []
    for i, r in enumerate(records, 1):
        create_time = r.get("createTime", "未知")
        page_code = r.get("pageCode", "未知")
        error_msg = r.get("blockedErrorMessage", "")
        lines.append(f"【受阻{i}】时间：{create_time}｜页面：{page_code}｜错误信息：{error_msg}")
    return "\n".join(lines)


def format_ubt_records(records: list[dict]) -> str:
    """格式化订单详情页浏览埋点为 prompt 文本。每条含 viewed_at / trigger / stay_ms / clicked_modules。"""
    if not records:
        return "无"
    lines = []
    for i, r in enumerate(records, 1):
        viewed_at = r.get("viewed_at", "未知")
        stay_sec = (r.get("stay_ms") or 0) / 1000
        trigger = r.get("trigger") or "未知"
        clicks = r.get("clicked_modules") or []
        click_text = "、".join(clicks) if clicks else "无点击"
        lines.append(
            f"【浏览{i}】时间：{viewed_at}｜停留：{stay_sec:.1f}s｜离开方式：{trigger}｜点击：{click_text}"
        )
    return "\n".join(lines)


def format_order_status(order_statuses: list[dict]) -> str:
    """格式化订单状态（退票/改签/航变）为精简 prompt 文本"""
    if not order_statuses:
        return ""
    lines = []
    for status in order_statuses:
        overview = status.get("订单概览", {})
        order_id = overview.get("订单号", "")
        order_state = overview.get("订单状态", "")
        desc = overview.get("行程描述", "")
        lines.append(f"订单{order_id} {desc} 状态：{order_state}")

        if overview.get("是否有未处理航变"):
            lines.append("  ⚠ 有未处理航变")
        if overview.get("不可改签原因"):
            lines.append(f"  不可改签：{overview['不可改签原因']}")
        if overview.get("不可退票原因"):
            lines.append(f"  不可退票：{overview['不可退票原因']}")

        for r in status.get("退票单列表", []):
            lines.append(f"  退票：{r.get('状态')}｜{r.get('退票类型', '')}｜乘客：{r.get('乘客', '')}｜航段：{r.get('航段', '')}｜申请时间：{r.get('申请时间', '')}")

        for r in status.get("改签单列表", []):
            lines.append(f"  改签：{r.get('状态')}｜{r.get('改签类型', '')}｜{r.get('改签原因', '')}｜申请时间：{r.get('申请时间', '')}")
            if r.get("失败原因"):
                lines.append(f"    失败原因：{r['失败原因']}")

        for c in status.get("航变列表", []):
            lines.append(f"  航变：{c.get('处理状态')}｜{c.get('航变原因', '')}｜原航班{c.get('原航班', '')} {c.get('原出发', '')} {c.get('原起飞', '')}→保护{c.get('保护航班', '')} {c.get('保护起飞', '')}")

    return "\n".join(lines)


def format_notification_timeline(notifications: list[dict]) -> str:
    """格式化去重后的通知时间线为 prompt 文本"""
    if not notifications:
        return ""
    lines = []
    for i, n in enumerate(notifications, 1):
        time_str = n.get("time", "未知")
        title = n.get("title", "")
        content = n.get("content", "")
        if content and content != title:
            lines.append(f"【通知{i}】{time_str}｜{title}｜{content}")
        else:
            lines.append(f"【通知{i}】{time_str}｜{title}")
    return "\n".join(lines)


def format_complaints(complaints: list[dict]) -> str:
    """格式化投诉单为 prompt 文本，区分活跃与已结案状态"""
    if not complaints:
        return ""
    lines = []
    for i, c in enumerate(complaints, 1):
        status = c.get("状态", "")
        deadline = c.get("截止时间", "")
        deadline_str = f"｜截止：{deadline}" if deadline else ""
        create_time = c.get("创建时间", "")
        create_str = f"｜创建：{create_time}" if create_time else ""
        lines.append(
            f"【投诉{i}】{status}｜{c.get('类别', '')}-{c.get('原因', '')}{deadline_str}{create_str}\n"
            f"  内容：{c.get('内容', '')}"
        )
    return "\n".join(lines)


def format_implus_messages(messages: list[dict]) -> str:
    """格式化 IMPlus 最近对话为 prompt 文本"""
    if not messages:
        return ""
    lines = []
    for m in messages:
        time_str = m.get("时间", "")
        speaker = m.get("发言人", "")
        content = m.get("内容", "")
        lines.append(f"[{time_str}] {speaker}：{content}")
    return "\n".join(lines)


def format_tts_records(records: list[dict]) -> str:
    """格式化TTS外呼记录为 prompt 文本"""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records, 1):
        time_str = r.get("time", "未知")
        content = r.get("content", "")
        call_result = r.get("callResult", "")
        biz_type = r.get("businessType", "")
        lines.append(
            f"【外呼{i}】{time_str}｜类型：{biz_type}｜接听状态：{call_result}\n"
            f"  通知内容：{content}"
        )
    return "\n".join(lines)


# ---- 多语言输出指令 ----

# 本地兜底映射，当 QConfig 不可用时使用
_DEFAULT_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "zh": "Chinese (中文)",
    "th": "Thai (ภาษาไทย)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "ms": "Malay (Bahasa Melayu)",
    "de": "German (Deutsch)",
    "fr": "French (Français)",
}


def _get_language_names() -> dict[str, str]:
    """从 QConfig 动态读取语言名称映射（支持热更新）。读不到则回退到本地默认值。"""
    return _get_config_value("language_names", _DEFAULT_LANGUAGE_NAMES)


def build_language_instruction(language: str | None) -> str:
    """构建多语言输出指令后缀，追加到 system prompt 末尾。
    为空或 'zh' 时返回空字符串，不影响原有逻辑。
    语言名称映射从 QConfig 动态读取，支持热更新。
    """
    if not language or language.lower() == "zh":
        return ""

    language_names = _get_language_names()
    lang_name = language_names.get(language.lower(), language)

    return f"""

## 语言要求
你必须将最终输出的所有 JSON 字段值翻译为 {lang_name}。具体包括：
- "预测分析"：用 {lang_name} 撰写推理过程
- "预期提示一级"：将标签体系中的一级分类翻译为 {lang_name}
- "预期提示二级"：将标签体系中的二级分类翻译为 {lang_name}
- "预期提示事件小结"：用 {lang_name} 撰写事件小结
- "辅助话术"：用 {lang_name} 撰写，符合该语言文化下的客服沟通习惯和礼貌用语

注意：JSON 的 key 名（如"预测分析"、"辅助话术"等）保持中文不变，仅翻译 value 值。"置信度"仍为整数，无需翻译。"""


# ---- 批量评估 Prompt ----

_DEFAULT_EVALUATION_SYSTEM_PROMPT = """\
你是一个意图预测准确性评估专家。你将收到三部分信息：
1. 我们在用户进线前做出的意图预测（包含标签、事件小结和辅助话术）
2. 预测时模型可用的全部信息（即用户Prompt，包含订单状态、通知记录、投诉单、在线会话等）
3. 该次进线的完整客服-用户对话记录

你的任务：
1. 根据对话内容，生成一句话小结（概括用户本次进线的核心诉求，50字以内）
2. 判断我们的预测是否准确
3. 分析基于预测时可用信息，该真实意图是否有可能被预测出来
4. 判断辅助话术的方向是否与用户真实意图的一级方向一致

## 核心判断原则

预测的对象是"用户为什么进线"——即用户进线的核心动机/意图。判断标准如下：

【准确】的情况：
- 预测的核心意图与用户进线的实际动机一致（如预测"催退票"，用户确实是为退票相关事宜进线）
- 预测方向正确但细节有偏差（如预测"改签咨询"实际是"改签操作"）
- 用户在对话中附带了情绪表达、投诉、额外要求，但核心诉求与预测一致
- 对话中客服主动处理了预测涉及的事项，即使用户没有显式开口要求

【不准确】的情况：
- 预测的核心意图与用户实际诉求明显不同（如预测"退票"实际是"改签"）
- 用户进线的真实原因是一个完全不同的事项

## 特别注意
- 不要因为对话过程中的附带行为（如用户抱怨、要求换沟通方式、简短确认等）而否定预测的准确性
- 关注"用户为什么来"而不是"对话中发生了什么"
- 如果对话内容显示客服在处理预测所涉及的事项，这本身就说明预测是准确的

## 可预测性分析

在判断准确性的同时，你还需要回溯分析：基于【预测时可用信息】中提供的所有数据（订单状态、历史交互摘要、系统通知记录、IVR外呼通知、投诉单、在线会话记录、用户受阻记录、订单详情页浏览记录），模型是否有可能预测出用户的真实进线意图。

判断标准：
- isPredictable = 1（可预测）：可用信息中已有明确信号直接指向真实意图，模型应当能预测正确
- isPredictable = 2（部分可预测）：可用信息中有相关信号可推导出大方向，但细节信息不足以精确预测
- isPredictable = 0（不可预测）：可用信息中完全没有与真实意图相关的信号，属于信息盲区

输出要求：
- supportingSignals：列出可用信息中哪些具体信息片段支持预测出真实意图（多条用｜分隔），若无则留空字符串
- missingInfo：列出预测真实意图所需但可用信息中缺失的关键信息（多条用｜分隔），若无则留空字符串

## 辅助话术方向评估

判断我们生成的辅助话术所表达的一级事件方向，与用户真实进线意图的一级方向是否一致。

一级事件方向包括：出票、增值服务、报销凭证、改签、求助/投诉、航变、订单查询修改、退票、预订。

判断标准：
- isScriptDirectionAccurate = true：话术中提及的一级事件方向与用户真实进线的一级方向一致（如话术提到"退票相关"，用户确实是退票类需求）
- isScriptDirectionAccurate = false：话术中提及的一级事件方向与用户真实进线的一级方向不一致（如话术提到"增值服务"，用户实际是退票需求）

注意：只判断方向是否一致，不要求话术内容细节完全匹配。

你必须以 JSON 格式输出，不要输出其他内容：
{"summary": "对话小结(50字以内)", "isAccurate": true或false, "isPredictable": 1或2或0, "supportingSignals": "信号1｜信号2", "missingInfo": "缺失信息1｜缺失信息2", "isScriptDirectionAccurate": true或false}"""


def get_evaluation_system_prompt() -> str:
    """获取评估 system prompt - 优先 QConfig，回退到本地默认"""
    return _get_config_value(
        "prompt.evaluation_system",
        _DEFAULT_EVALUATION_SYSTEM_PROMPT,
        config_name="prompt_config.json",
    )


def build_evaluation_user_prompt(
    level1_label: str,
    level2_label: str,
    event_summary: str,
    prediction_analysis: str,
    conversation: str,
    user_prompt: str = "",
    suggested_script: str = "",
) -> str:
    """构建评估用户 prompt（含预测时可用信息，用于可预测性分析和话术方向评估）"""
    parts = [
        f"【我们的预测】\n"
        f"- 一级标签: {level1_label}\n"
        f"- 二级标签: {level2_label}\n"
        f"- 事件小结: {event_summary}\n"
        f"- 预测分析: {prediction_analysis}\n"
        f"- 辅助话术: {suggested_script}",
    ]
    if user_prompt:
        parts.append(f"【预测时可用信息（用户Prompt）】\n{user_prompt}")
    parts.append(f"【本次进线对话记录】\n{conversation}")
    return "\n\n".join(parts)

