"""
订单详情页埋点 (UBT) 客户端
通过 MyTrix HTTP SQL 引擎查询 Hive 表 dwflt.edw_ubt_trace_order_detail_main_page_di。
只取当前订单、进线当天、按 pvid 去重后的最近 N 次浏览，并做字段瘦身避免污染 LLM 上下文。
"""
import asyncio
import json
from datetime import date

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


_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent_api_calls)
    return _semaphore


def _build_sql(order_id: int, today: str) -> str:
    # order_id 已由 Pydantic 强制为 int，today 由本地 strftime 生成，无注入风险
    return (
        "select pvid, start_time, data_info "
        "from dwflt.edw_ubt_trace_order_detail_main_page_di "
        f"where d = '{today}' "
        f"  and order_id = '{int(order_id)}' "
        "order by start_time desc"
    )


def _call_mytrix(sql: str) -> dict:
    resp = requests.post(
        settings.ubt_mytrix_url,
        json={
            "sql": sql,
            "employeeId": settings.ubt_employee_id,
            "rawSql": True,
        },
        headers={"Content-Type": "application/json"},
        timeout=settings.api_timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_clicked_modules(show_click_raw) -> list[str]:
    """从 Show_Click 字符串/列表中抽取 click=1 的条目，格式 '{module}/{title}' 或 '{module}/{title}_{type}'。"""
    try:
        items = json.loads(show_click_raw) if isinstance(show_click_raw, str) else show_click_raw
    except (ValueError, TypeError):
        return []
    if not isinstance(items, list):
        return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("click") != 1:
            continue
        module = item.get("module", "") or ""
        title = item.get("title", "") or ""
        type_ = item.get("type", "") or ""
        label = f"{module}/{title}" if module or title else ""
        if type_:
            label = f"{label}_{type_}" if label else type_
        if label:
            result.append(label)
    return result


def _shrink(row: dict) -> dict:
    """将单条 Hive 行瘦身为 4 字段 dict：viewed_at / trigger / stay_ms / clicked_modules。"""
    try:
        data_info_raw = row.get("data_info", "") or ""
        data_info = json.loads(data_info_raw) if isinstance(data_info_raw, str) and data_info_raw else {}
    except (ValueError, TypeError):
        data_info = {}

    try:
        stay_ms = int(data_info.get("CostTime", 0) or 0)
    except (ValueError, TypeError):
        stay_ms = 0

    return {
        "viewed_at": row.get("start_time", ""),
        "trigger": data_info.get("TriggerType", "") or "",
        "stay_ms": stay_ms,
        "clicked_modules": _extract_clicked_modules(data_info.get("Show_Click", "[]")),
    }


def _dedup_and_limit(rows: list[dict], limit: int) -> list[dict]:
    """按 pvid 去重，每个 pvid 保留 start_time 最大的一条；再按 start_time DESC 取前 limit 条。"""
    by_pvid: dict = {}
    for row in rows:
        pvid = row.get("pvid")
        if pvid is None:
            continue
        existing = by_pvid.get(pvid)
        if existing is None or (row.get("start_time", "") > existing.get("start_time", "")):
            by_pvid[pvid] = row
    sorted_rows = sorted(by_pvid.values(), key=lambda r: r.get("start_time", ""), reverse=True)
    return sorted_rows[:limit]


async def query_order_detail_ubt(order_id: int, limit: int | None = None) -> list[dict]:
    """
    查询当前订单当天的订单详情页浏览埋点。
    - 按 pvid 去重，每次页面访问保留 1 条
    - 取最近 limit 次浏览（默认 settings.ubt_recent_count）
    - 每条瘦身为 {viewed_at, trigger, stay_ms, clicked_modules}
    失败（网络/解析/上游异常）一律返空列表，不抛异常、不阻断主流程。
    """
    effective_limit = limit if limit is not None else settings.ubt_recent_count
    today = date.today().strftime("%Y-%m-%d")
    sql = _build_sql(order_id, today)

    try:
        async with _get_semaphore():
            resp = await asyncio.to_thread(_call_mytrix, sql)
    except Exception as e:
        LogHelper.log_warn(f"查询订单详情页埋点失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []

    rows = resp.get("data") if isinstance(resp, dict) else None
    if not isinstance(rows, list) or not rows:
        return []

    top_rows = _dedup_and_limit(rows, effective_limit)
    return [_shrink(r) for r in top_rows]
