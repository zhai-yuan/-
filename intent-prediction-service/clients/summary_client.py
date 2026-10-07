"""
SOA API 客户端封装
封装 5 个 SOA 接口: 关联订单 / 进线记录 / 外呼记录 / 电话摘要 / 会话摘要 / 用户受阻
所有函数为 async，内部通过 asyncio.to_thread 执行阻塞的 requests 调用
"""
import asyncio
import uuid

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# 并发信号量 - 防止同时打爆下游
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent_api_calls)
    return _semaphore


def _call_api(url: str, operation: str, payload: dict) -> dict:
    """同步 SOA 接口调用"""
    resp = requests.post(
        f"{url}/{operation}",
        json=payload,
        timeout=settings.api_timeout,
    )
    resp.raise_for_status()
    return resp.json()


async def _async_call_api(url: str, operation: str, payload: dict) -> dict:
    """异步包装: 带信号量控制"""
    async with _get_semaphore():
        return await asyncio.to_thread(_call_api, url, operation, payload)


async def get_related_order_ids(order_id: int) -> tuple[set[int], dict]:
    """通过 API 11817 反查关联订单，返回 (order_id集合, 原始响应) 供订单状态提取复用"""
    payload = {
        "AppID": 100066134,
        "TransactionId": str(uuid.uuid4()),
        "OpenOrderDetailSearchItems": [
            {
                "OrderID": order_id,
                "Language": 0,
                "ResponseSurface": False,
                "NeedMergeInsuranceAccount": False,
                "PenaltyShowType": 1,
                "ShowMultiCurrency": False,
                "ShowRange": 0,
            }
        ],
    }
    try:
        data = await _async_call_api(settings.api_order_detail, "OpenOrderDetailSearch", payload)
    except Exception as e:
        LogHelper.log_warn(f"反查关联订单失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return {order_id}, {}

    all_order_ids = {order_id}
    result_items = data.get("OpenOrderDetailSearchResultItem", [])
    if isinstance(result_items, dict):
        result_items = [result_items]
    for item in result_items:
        basic = item.get("BasicOrderInformation", {})
        related_list = basic.get("RelatedOrders", [])
        if isinstance(related_list, dict):
            related_list = [related_list]
        for rel in related_list:
            rid = rel.get("RelatedOrderId")
            if rid:
                all_order_ids.add(int(rid))

    return all_order_ids, data


async def get_incomecall_records(order_id: int, start_time: str, end_time: str) -> list[dict]:
    """通过 API 14537 获取进线记录"""
    payload = {"orderId": order_id, "startTime": start_time, "endTime": end_time}
    try:
        data = await _async_call_api(settings.api_calling_center, "getIncomecallRecord", payload)
        return data.get("incomecallRecords", []) or []
    except Exception as e:
        LogHelper.log_warn(f"获取进线记录失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []


async def get_outcall_records(order_id: int, start_time: str, end_time: str) -> list[dict]:
    """通过 API 14537 获取外呼记录"""
    payload = {"orderId": order_id, "startTime": start_time, "endTime": end_time}
    try:
        data = await _async_call_api(settings.api_calling_center, "getOutcallRecord", payload)
        return data.get("outcallRecords", []) or []
    except Exception as e:
        LogHelper.log_warn(f"获取外呼记录失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []


async def query_tel_summary(call_id: str) -> list[dict]:
    """通过 API 17307 查询电话摘要"""
    payload = {"requestId": str(uuid.uuid4()), "callId": call_id}
    try:
        data = await _async_call_api(settings.api_tel_summary, "queryAISummary", payload)
        return data.get("summaryList", []) or []
    except Exception as e:
        LogHelper.log_warn(f"查询电话摘要失败 callId={call_id}: {e}", {"callId": call_id})
        return []


async def query_chat_summary(session_id: str) -> list[dict]:
    """通过 API 15408 查询在线会话摘要"""
    payload = {"requestId": str(uuid.uuid4()), "sessionId": session_id}
    try:
        data = await _async_call_api(settings.api_chat_summary, "queryChatAISummary", payload)
        return data.get("summaryList", []) or []
    except Exception as e:
        LogHelper.log_warn(f"查询会话摘要失败 sessionId={session_id}: {e}", {"sessionId": session_id})
        return []


async def query_user_blocked_records(order_id: int) -> list[dict]:
    """通过 API 11504 查询用户受阻记录"""
    payload = {"orderId": order_id}
    try:
        data = await _async_call_api(settings.api_user_behavior, "queryUserBehaviorRecords", payload)
        records = data.get("userBehaviorRecordList", []) or []
        return [r for r in records if r.get("blockedErrorMessage")]
    except Exception as e:
        LogHelper.log_warn(f"查询用户受阻记录失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []
