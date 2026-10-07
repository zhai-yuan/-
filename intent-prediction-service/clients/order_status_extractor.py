"""
订单状态提取器
从 OpenOrderDetailSearch 原始响应中提取退票/改签/航变核心状态
"""
from datetime import datetime


REBOOK_STATUS_MAP = {
    "U": "未提交", "W": "等待处理", "P": "处理中",
    "C": "已取消", "S": "改签成功", "F": "改签失败",
}

REBOOK_TYPE_MAP = {1: "同舱改期", 2: "升舱"}

REBOOK_REASON_MAP = {1: "自愿改签", 2: "航变改签"}

REFUND_VIEW_STATE_MAP = {
    "U": "未提交", "W": "等待处理", "P": "处理中",
    "C": "已取消", "S": "退票成功", "F": "退票失败",
}

REFUND_TYPE_CODE_MAP = {1: "自愿", 2: "非自愿"}

ORDER_STATUS_MAP = {
    "N": "新订单", "W": "等待处理", "S": "已出票",
    "R": "已退票", "C": "已取消", "P": "处理中",
}

CHANGE_STATUS_MAP = {"S": "已处理", "W": "待处理", "P": "处理中", "C": "已取消"}

RR_STATUS_MAP = {"A": "已接受", "R": "已改签", "F": "已退票", "W": "待确认", "N": "未处理"}


def _parse_date(date_str) -> str | None:
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


def _extract_refund_info(refund_list: list) -> list[dict]:
    results = []
    for item in refund_list:
        status = REFUND_VIEW_STATE_MAP.get(item.get("RefundOrderViewState"), item.get("RefundOrderViewState"))
        passengers = set()
        segments = []
        for d in item.get("RefundDetailInfoList", []):
            passengers.add(d.get("PassengerName", ""))
            segments.append(d.get("DportAportPair", ""))
        results.append({
            "退票单号": item.get("RefundOrderID"),
            "状态": status,
            "来源": item.get("Source"),
            "申请时间": _parse_date(item.get("RefundTime")),
            "乘客": "、".join(p for p in passengers if p),
            "航段": "、".join(dict.fromkeys(s for s in segments if s)),
            "退票类型": REFUND_TYPE_CODE_MAP.get(
                item.get("RefundDetailInfoList", [{}])[0].get("RefundTypeCode") if item.get("RefundDetailInfoList") else None,
                ""
            ),
        })
    return results


def _extract_rebook_info(rebook_list: list) -> list[dict]:
    results = []
    for item in rebook_list:
        status = REBOOK_STATUS_MAP.get(item.get("RebookStatus"), item.get("RebookStatus"))
        results.append({
            "改签单号": item.get("Rbkid"),
            "状态": status,
            "改签类型": REBOOK_TYPE_MAP.get(item.get("RebookType"), ""),
            "改签原因": REBOOK_REASON_MAP.get(item.get("RebookingReasonType"), ""),
            "申请时间": _parse_date(item.get("RebookingTime")),
            "完成时间": _parse_date(item.get("RebookFinishTime")),
            "失败原因": item.get("FailReason") or None,
        })
    return results


def _extract_flight_change_info(change_list: list) -> list[dict]:
    results = []
    for item in change_list:
        status = CHANGE_STATUS_MAP.get(item.get("ChangeStatus"), item.get("ChangeStatus"))
        rr_status = RR_STATUS_MAP.get(item.get("RRStatus"), item.get("RRStatus"))
        results.append({
            "航变单号": item.get("ChangeOrderID"),
            "处理状态": status,
            "旅客响应": rr_status,
            "航变原因": item.get("FlightChangeReason"),
            "乘客": item.get("Passengers"),
            "原航班": item.get("OriginFlight"),
            "原出发": f"{item.get('OriginDCityName', '')}-{item.get('OriginACityName', '')}",
            "原起飞": _parse_date(item.get("OriginDdate")),
            "保护航班": item.get("ProtectFlight"),
            "保护起飞": _parse_date(item.get("ProtectDdate")),
            "创建时间": _parse_date(item.get("CreateTime")),
        })
    return results


def extract_order_status(raw_response: dict) -> list[dict]:
    """从 OpenOrderDetailSearch 原始响应提取订单状态摘要"""
    result_items = raw_response.get("OpenOrderDetailSearchResultItem", [])
    if isinstance(result_items, dict):
        result_items = [result_items]

    all_results = []
    for item in result_items:
        basic = item.get("BasicOrderInformation", {})

        order_summary = {
            "订单号": basic.get("OrderID"),
            "订单状态": ORDER_STATUS_MAP.get(basic.get("OrderStatus"), basic.get("OrderStatus")),
            "行程描述": basic.get("OrderDesc"),
            "是否有航变": basic.get("HasFlightChange") == "T",
            "是否有未处理航变": basic.get("HasFlightChangeUntreated") == "T",
            "可否改签": basic.get("RebookAble", False),
            "不可改签原因": basic.get("UnrebookReason") or None,
            "可否退票": basic.get("Refundable", False),
            "不可退票原因": basic.get("UnrefundReason") or None,
        }

        refund_infos = _extract_refund_info(item.get("OrderRefundInfos", []))
        rebook_infos = _extract_rebook_info(item.get("OrderRebookInfos", []))
        change_infos = _extract_flight_change_info(item.get("ChangeOrderList", []))

        all_results.append({
            "订单概览": order_summary,
            "退票单列表": refund_infos,
            "改签单列表": rebook_infos,
            "航变列表": change_infos,
        })

    return all_results
