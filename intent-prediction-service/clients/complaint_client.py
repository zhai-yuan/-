"""
投诉单客户端
查询投诉单（含活跃和已完成，帮助LLM判断投诉是否仍在进行中）
"""
from datetime import datetime

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


def _parse_soa_date(date_str) -> str | None:
    if not date_str or not isinstance(date_str, str) or not date_str.startswith("/Date("):
        return None
    try:
        ts_part = date_str.replace("/Date(", "").replace(")/", "")
        ts_ms = int(ts_part.split("+")[0].split("-")[0])
        if ts_ms < 0:
            return None
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return None


async def search_complaints(order_id: int) -> list[dict]:
    """查询投诉单（含活跃和已完成），返回精简字段"""
    payload = {"orderID": str(order_id)}
    try:
        data = await _async_call_api(settings.api_complaint, "searchComplaint", payload)
    except Exception as e:
        LogHelper.log_warn(f"查询投诉单失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []

    results = []
    for c in data.get("complaints", []):
        status_name = c.get("statusName", "")
        if not status_name:
            continue
        content = c.get("content", "")
        if len(content) > 150:
            content = content[:150] + "..."
        results.append({
            "投诉单号": c.get("complaintID"),
            "状态": status_name,
            "类别": c.get("category", ""),
            "原因": c.get("reason", ""),
            "内容": content,
            "创建时间": _parse_soa_date(c.get("createDate")),
            "截止时间": _parse_soa_date(c.get("deadline")),
        })
    return results
