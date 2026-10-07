"""
批量准确性评估 - 核心逻辑
每天 T+1 将预测结果与会话小结进行匹配，用 LLM 判定预测准确性。
"""
import asyncio
import json
import re
import time
from datetime import date, timedelta

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
from config.prompts import get_evaluation_system_prompt, build_evaluation_user_prompt
from clients.hive_summary_client import check_partition_ready, query_conversations_by_session_ids
from app.db_helper import get_db_helper


def _build_evaluation_llm():
    """构建评估专用 LLM 主实例（vLLM 自部署 Qwen3.5-27B-FP8）"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.evaluation_llm_model,
        base_url=settings.evaluation_llm_base_url,
        api_key=settings.evaluation_llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=2,
        max_tokens=8192,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )


def _build_evaluation_llm_fallback():
    """构建评估 fallback LLM 实例（aigw Qwen3.5-27B）"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_fallback_model,
        base_url=settings.llm_fallback_base_url,
        api_key=settings.llm_fallback_api_key,
        timeout=settings.llm_timeout,
        max_retries=2,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )


def _parse_evaluation_result(raw: str) -> dict | None:
    """
    从 LLM 输出中提取评估结果 JSON。
    返回 {"summary": str, "isAccurate": bool, "isPredictable": int|None,
           "supportingSignals": str|None, "missingInfo": str|None} 或 None（解析失败）。
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
                return None
        else:
            return None

    # 验证必要字段
    if "isAccurate" not in result:
        return None
    if not isinstance(result["isAccurate"], bool):
        # 尝试转换
        val = result["isAccurate"]
        if isinstance(val, str):
            result["isAccurate"] = val.strip().lower() in ("true", "1", "yes")
        elif isinstance(val, (int, float)):
            result["isAccurate"] = bool(val)
        else:
            return None

    # 确保 summary 字段存在
    if "summary" not in result:
        result["summary"] = ""

    # 解析可预测性字段（容错：缺失时给 None，不阻塞主流程）
    is_predictable = result.get("isPredictable")
    if is_predictable is not None:
        try:
            is_predictable = int(is_predictable)
            if is_predictable not in (0, 1, 2):
                is_predictable = None
        except (ValueError, TypeError):
            is_predictable = None
    result["isPredictable"] = is_predictable

    result.setdefault("supportingSignals", "")
    result.setdefault("missingInfo", "")
    # 确保是字符串
    if not isinstance(result["supportingSignals"], str):
        result["supportingSignals"] = str(result["supportingSignals"]) if result["supportingSignals"] else ""
    if not isinstance(result["missingInfo"], str):
        result["missingInfo"] = str(result["missingInfo"]) if result["missingInfo"] else ""

    # 解析辅助话术方向准确性字段（容错：缺失时给 None）
    is_script_direction = result.get("isScriptDirectionAccurate")
    if is_script_direction is not None:
        if isinstance(is_script_direction, bool):
            pass  # already bool
        elif isinstance(is_script_direction, str):
            is_script_direction = is_script_direction.strip().lower() in ("true", "1", "yes")
        elif isinstance(is_script_direction, (int, float)):
            is_script_direction = bool(is_script_direction)
        else:
            is_script_direction = None
    result["isScriptDirectionAccurate"] = is_script_direction

    return result


async def _evaluate_single(
    llm,
    llm_fallback,
    prediction: dict,
    conversation: str,
    semaphore: asyncio.Semaphore,
    db_helper,
) -> bool:
    """
    评估单条预测记录。主模型失败时自动 fallback。
    返回 True 表示成功处理，False 表示失败跳过。
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    session_id = prediction.get("sessionId", "")
    system_prompt = get_evaluation_system_prompt()
    user_prompt = build_evaluation_user_prompt(
        level1_label=prediction.get("level1Label", ""),
        level2_label=prediction.get("level2Label", ""),
        event_summary=prediction.get("eventSummary", ""),
        prediction_analysis=prediction.get("predictionAnalysis", ""),
        conversation=conversation,
        user_prompt=prediction.get("userPrompt", ""),
        suggested_script=prediction.get("suggestedScript", ""),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = None

    # 尝试主模型
    try:
        async with semaphore:
            response = await asyncio.to_thread(llm.invoke, messages)
    except Exception as e:
        LogHelper.log_warn(f"评估主模型调用失败 sessionId={session_id}: {e}，尝试 fallback")

    # 主模型失败或返回为空，尝试 fallback
    if response is None:
        try:
            async with semaphore:
                response = await asyncio.to_thread(llm_fallback.invoke, messages)
        except Exception as e:
            LogHelper.log_warn(f"评估 fallback 模型也失败 sessionId={session_id}: {e}")
            return False

    # 解析结果
    result = _parse_evaluation_result(response.content)
    if result is None:
        LogHelper.log_warn(
            f"评估结果解析失败 sessionId={session_id}, raw={response.content[:200]}"
        )
        return False

    # 写回 MySQL（summary 落到 callSummary 字段，可预测性分析落到新增字段）
    success = db_helper.update_evaluation_result(
        session_id=session_id,
        is_accurate=result["isAccurate"],
        call_summary=result["summary"],
        is_predictable=result.get("isPredictable"),
        supporting_signals=result.get("supportingSignals", ""),
        missing_info=result.get("missingInfo", ""),
        is_script_direction_accurate=result.get("isScriptDirectionAccurate"),
    )
    return success


async def preflight_check(target_date: str | None = None) -> tuple[bool, str]:
    """
    前置检查：验证批量评估的前提条件是否满足。
    快速返回（秒级），用于 Zeus 调度场景的失败快速反馈。
    返回 (ok: bool, message: str)
    """
    from app.db_helper import get_db_helper

    # 确定目标日期
    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # 检查 1: 是否有未评估的预测记录
    db_helper = get_db_helper()
    predictions = db_helper.query_predictions_by_date(target_date)
    if not predictions:
        return True, f"无未评估的预测记录 (date={target_date})，无需执行"

    # 检查 2: Hive 分区是否就绪
    if not check_partition_ready(target_date):
        return False, f"Hive 分区数据未就绪 (date={target_date})，等待 Zeus 重试"

    return True, f"前置检查通过: {len(predictions)} 条待评估记录"


async def run_batch_evaluation(target_date: str | None = None, limit: int | None = None):
    """
    执行批量评估主流程。
    target_date: 指定评估日期 (YYYY-MM-DD)，默认为前一天。
    limit: 最多处理的记录数，None 表示不限制（全量处理）。
    返回 (success: bool, stats: dict)
    """
    start_time = time.time()

    # 确定目标日期
    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    LogHelper.log_info(
        f"批量评估任务开始: target_date={target_date}"
        + (f", limit={limit}" if limit else "")
    )

    # 1. 从 MySQL 拉取未评估的预测记录
    db_helper = get_db_helper()
    predictions = db_helper.query_predictions_by_date(target_date)
    if not predictions:
        LogHelper.log_info(f"无未评估的预测记录 (date={target_date})，任务结束")
        return True, {"total_predictions": 0, "reason": "no_predictions"}

    # 应用 limit 截断
    if limit and limit > 0:
        predictions = predictions[:limit]

    # 2. 检查 Hive 分区是否就绪
    if not check_partition_ready(target_date):
        LogHelper.log_warn(f"Hive 分区数据未就绪 d={target_date}")
        return False, {"total_predictions": len(predictions), "reason": "hive_not_ready"}

    # 3. 提取 sessionId 列表，查询 Hive 会话数据
    session_ids = [p.get("sessionId", "") for p in predictions if p.get("sessionId")]
    conversations = query_conversations_by_session_ids(target_date, session_ids)

    if not conversations:
        LogHelper.log_warn(f"Hive 中未匹配到任何会话数据 (date={target_date})")
        return False, {
            "total_predictions": len(predictions),
            "reason": "no_conversations_matched",
        }

    # 4. 内存匹配：只保留有会话数据的预测记录
    matched_predictions = [
        (p, conversations[p["sessionId"]])
        for p in predictions
        if p.get("sessionId") in conversations
    ]
    unmatched_session_ids = [
        p.get("sessionId", "")
        for p in predictions
        if p.get("sessionId") and p["sessionId"] not in conversations
    ]

    LogHelper.log_info(
        f"匹配完成: 预测记录 {len(predictions)} 条, "
        f"匹配到小结 {len(matched_predictions)} 条, "
        f"未匹配 {len(unmatched_session_ids)} 条"
    )

    # 5. 构建 LLM 实例，逐条评估
    llm = _build_evaluation_llm()
    llm_fallback = _build_evaluation_llm_fallback()
    semaphore = asyncio.Semaphore(settings.evaluation_concurrency)

    tasks = [
        _evaluate_single(llm, llm_fallback, pred, conversation, semaphore, db_helper)
        for pred, conversation in matched_predictions
    ]
    results = await asyncio.gather(*tasks)

    # 6. 统计
    success_count = sum(1 for r in results if r)
    failed_count = sum(1 for r in results if not r)
    duration = round(time.time() - start_time, 1)

    stats = {
        "total_predictions": len(predictions),
        "matched_count": len(matched_predictions),
        "unmatched_count": len(unmatched_session_ids),
        "unmatched_session_ids": unmatched_session_ids,
        "evaluated_count": success_count,
        "failed_count": failed_count,
        "duration_seconds": duration,
    }

    LogHelper.log_info(
        f"批量评估任务完成: {json.dumps(stats, ensure_ascii=False)}"
    )

    return True, stats
