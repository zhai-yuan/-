"""
LLM 客户端封装
- LangChain ChatOpenAI + fallback 兜底模型
- 输出解析与标签校验
"""
import json
import re

try:
    from bicommonkits.flight.bi.rm.common import LogHelper
except ImportError:
    import logging
    class LogHelper:
        _logger = logging.getLogger("intent-prediction")
        @staticmethod
        def log_info(msg, tag=None): LogHelper._logger.info(msg)
        @staticmethod
        def log_warn(msg, tag=None): LogHelper._logger.warning(msg)
        @staticmethod
        def log_error(msg, tag=None): LogHelper._logger.error(msg)

from config.settings import settings


_llm_chain = None


def _build_llm():
    """构建带 fallback 的 LLM 链"""
    from langchain_openai import ChatOpenAI

    primary = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )

    fallback = ChatOpenAI(
        model=settings.llm_fallback_model,
        base_url=settings.llm_fallback_base_url,
        api_key=settings.llm_fallback_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    return primary.with_fallbacks([fallback])


def get_llm():
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = _build_llm()
    return _llm_chain


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    调用 LLM 并返回完整文本。
    主模型失败时自动 fallback 到兜底模型。
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        LogHelper.log_error(f"LLM 调用最终失败 (主模型 + 兜底模型均失败): {e}")
        raise


def parse_prediction(raw: str, label_mapping: dict | None = None, language: str | None = None) -> dict:
    """
    从 LLM 输出中提取 JSON 结果。
    处理: markdown代码块包裹、前后缀文本、标签格式清洗、模糊匹配、置信度解析。
    当 language 指定非中文时，跳过标签模糊匹配（翻译后的标签无法与中文 taxonomy 对比）。
    """
    cleaned = raw.strip()

    # 处理 markdown ```json ... ``` 包裹
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    # 尝试直接解析
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试提取 {...} 子串
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                return {"raw_output": raw}
        else:
            return {"raw_output": raw}

    # 清洗一级标签格式 - 去除 【】[]
    if "预期提示一级" in result:
        result["预期提示一级"] = re.sub(r"[【】\[\]]", "", result["预期提示一级"])

    # 解析置信度字段 - 确保为整数
    if "置信度" in result:
        try:
            result["置信度"] = int(result["置信度"])
        except (ValueError, TypeError):
            result["置信度"] = 0
    else:
        # 模型未输出置信度时，默认给 50（中等）
        result["置信度"] = 50

    # 标签校验 - 模糊匹配（跳过"无法预测"标记）
    # 非中文输出时跳过标签匹配（翻译后的标签无法与中文 taxonomy 对比）
    is_chinese_output = not language or language.lower() == "zh"
    if is_chinese_output and label_mapping and "预期提示一级" in result:
        level1 = result["预期提示一级"]
        if level1 == "无法预测":
            pass  # 无法预测是合法输出，不做标签匹配
        elif level1 not in label_mapping:
            # 去除标点后再做包含匹配
            level1_clean = re.sub(r"[，、,\s]", "", level1)
            for key in label_mapping:
                key_clean = re.sub(r"[，、,\s]", "", key)
                if key_clean in level1_clean or level1_clean in key_clean or key in level1 or level1 in key:
                    result["预期提示一级"] = key
                    break
            else:
                LogHelper.log_warn(f"一级标签 '{level1}' 不在标签体系中，保留原始值")

    return result
