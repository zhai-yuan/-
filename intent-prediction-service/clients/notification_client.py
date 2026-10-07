"""
通知时间线客户端
合并 站内信 + 微信 + APP推送 三渠道，去重后返回统一通知列表
"""
import asyncio
from datetime import datetime, timedelta

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


# 无预测价值的模板话术，从内容中剔除
_BOILERPLATE_PHRASES = [
    "您可以通过APP订单详情页实时查看处理进度。",
    "您可点击进入机票详情页面实时查询订单状态。",
    "请耐心等待，",
    "为此我们深表歉意，您可登录携程app",
]


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


def _clean_content(text: str, max_chars: int) -> str:
    if not text:
        return ""
    for phrase in _BOILERPLATE_PHRASES:
        text = text.replace(phrase, "")
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _normalize_letter_history(data: dict, max_chars: int) -> list[dict]:
    """标准化站内信响应"""
    items = []
    for h in data.get("histories", []):
        content = ""
        for c in h.get("contents", []):
            if c.get("channelType") == "APP":
                content = c.get("content", "")
                break
        if not content and h.get("contents"):
            content = h["contents"][0].get("content", "")

        items.append({
            "time": h.get("sendTime", ""),
            "title": h.get("title", ""),
            "content": _clean_content(content, max_chars),
            "messageCode": h.get("messageCode", ""),
            "requestId": h.get("requestId", ""),
            "source": "站内信",
        })
    return items


def _normalize_wechat(data: dict, max_chars: int) -> list[dict]:
    """标准化微信响应"""
    items = []
    for m in data.get("messageList", []):
        content = ""
        for c in m.get("messageContentList", []):
            content = c.get("content", "")
            break

        time_str = _parse_soa_date(m.get("createTime")) or ""
        items.append({
            "time": time_str,
            "title": m.get("messageName", ""),
            "content": _clean_content(content, max_chars),
            "messageCode": m.get("messageCode", ""),
            "requestId": "",
            "source": "微信",
        })
    return items


def _normalize_app_push(data: dict, max_chars: int) -> list[dict]:
    """标准化APP推送响应"""
    items = []
    for a in data.get("appHistoryList", []):
        time_str = _parse_soa_date(a.get("dataChangeLastTime")) or ""
        items.append({
            "time": time_str,
            "title": a.get("content", ""),
            "content": a.get("messagename", ""),
            "messageCode": a.get("messagecode", ""),
            "requestId": a.get("requestid", ""),
            "source": "APP推送",
        })
    return items


def _deduplicate(letters: list, wechats: list, app_pushes: list, max_count: int) -> list[dict]:
    """
    跨渠道去重，优先级：站内信 > 微信 > APP推送
    去重key：requestId (精确) 或 (messageCode, 5分钟时间桶)
    """
    seen_request_ids = set()
    seen_code_time = set()
    result = []

    for source_list in [letters, wechats, app_pushes]:
        for item in source_list:
            request_id = item.get("requestId")
            if request_id:
                if request_id in seen_request_ids:
                    continue
                seen_request_ids.add(request_id)
            else:
                code = item.get("messageCode", "")
                time_bucket = item.get("time", "")[:15]
                key = (code, time_bucket)
                if key in seen_code_time:
                    continue
                seen_code_time.add(key)
            result.append(item)

    result.sort(key=lambda x: x.get("time", ""), reverse=True)
    return result[:max_count]


async def _query_letter_history(order_id: int, start_time: str, end_time: str) -> dict:
    payload = {
        "queryType": "ORDER_ID",
        "queryValue": str(order_id),
        "startDate": start_time,
        "endDate": end_time,
        "pageIndex": 1,
        "pageSize": 20,
        "sortBySendTime": "DESC",
    }
    try:
        return await _async_call_api(settings.api_notification, "queryLetterHistoryV2", payload)
    except Exception as e:
        LogHelper.log_warn(f"查询站内信失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return {}


async def _query_wechat_messages(order_id: int) -> dict:
    payload = {"MessageType": "WECHAT", "OrderID": order_id}
    try:
        return await _async_call_api(settings.api_wechat, "queryMessage", payload)
    except Exception as e:
        LogHelper.log_warn(f"查询微信消息失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return {}


async def _query_app_push_history(order_id: int, start_time: str, end_time: str) -> dict:
    payload = {
        "orderID": order_id,
        "startDate": start_time,
        "endDate": end_time,
        "pageSize": 20,
        "sortType": "DESC",
    }
    try:
        return await _async_call_api(settings.api_notification, "queryAppHistory", payload)
    except Exception as e:
        LogHelper.log_warn(f"查询APP推送失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return {}


async def get_notification_timeline(order_id: int, start_time: str, end_time: str) -> list[dict]:
    """并行获取三渠道通知，去重合并为统一时间线"""
    letter_data, wechat_data, push_data = await asyncio.gather(
        _query_letter_history(order_id, start_time, end_time),
        _query_wechat_messages(order_id),
        _query_app_push_history(order_id, start_time, end_time),
    )

    max_chars = settings.notification_content_max_chars
    letters = _normalize_letter_history(letter_data, max_chars)
    wechats = _normalize_wechat(wechat_data, max_chars)
    app_pushes = _normalize_app_push(push_data, max_chars)

    return _deduplicate(letters, wechats, app_pushes, settings.notification_max_count)
