"""
机票外呼TTS结果客户端
查询订单的IVR外呼通知内容（航变/退票/改签等TTS语音通知）
"""
import json

try:
    from bicommonkits.flight.bi.rm.common import LogHelper
except ImportError:
    import logging
    class LogHelper:
        _logger = logging.getLogger("intent-prediction")
        @staticmethod
        def log_warn(msg, tag=None): LogHelper._logger.warning(msg)

from clients.summary_client import _async_call_api
from config.settings import settings


def _extract_tts_content(detail: dict) -> str:
    """从 DetailOutCallResult.Content 中提取 TTS 文本"""
    raw_content = detail.get("Content", "")
    if not raw_content:
        return ""
    try:
        outer = json.loads(raw_content)
        inner_content = outer.get("content", "")
        if not inner_content:
            return ""
        inner = json.loads(inner_content)
        parts = []
        for key in ("TTS1", "TTS2", "TTS3"):
            val = inner.get(key, "")
            if val and val.strip():
                parts.append(val.strip())
        return "".join(parts)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def _normalize_tts_result(item: dict) -> dict | None:
    """将单条外呼结果标准化为内部格式"""
    detail = item.get("DetailOutCallResult") or {}
    tts_text = _extract_tts_content(detail)
    if not tts_text:
        return None

    call_result = detail.get("CallResult", "")
    result_map = {"A": "已接听", "B": "未接听", "C": "无法接通"}
    result_desc = result_map.get(call_result, call_result)

    return {
        "time": item.get("CreateTime", ""),
        "content": tts_text,
        "callResult": result_desc,
        "businessType": "改签" if item.get("BusinessType") == 1 else "退票",
    }


async def get_tts_outcall_results(order_id: int) -> list[dict]:
    """查询机票外呼TTS结果，返回标准化列表"""
    payload = {
        "OrderId": order_id,
        "BusinessId": 0,
        "BusinessType": 0,
        "FlightOutCallTaskId": 0,
        "NeedDetail": True,
    }
    try:
        data = await _async_call_api(
            settings.api_tts_outcall, "QueryFlightOutCallResult", payload
        )
    except Exception as e:
        LogHelper.log_warn(
            f"查询TTS外呼结果失败 order_id={order_id}: {e}",
            {"orderId": str(order_id)},
        )
        return []

    items = data.get("OutCallResultList") or []
    results = []
    for item in items:
        normalized = _normalize_tts_result(item)
        if normalized:
            results.append(normalized)

    results.sort(key=lambda x: x.get("time", ""), reverse=True)
    return results[:settings.tts_max_count]
