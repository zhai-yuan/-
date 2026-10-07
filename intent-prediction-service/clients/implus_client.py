"""
IMPlus 在线会话客户端
获取订单相关的在线会话消息，仅提取有效文本对话
"""
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


async def get_implus_messages(order_id: int) -> list[dict]:
    """
    获取 IMPlus 最近会话的文本对话
    仅返回 msgtype==0 的纯文本消息，排除系统消息，取最近一个 session
    """
    payload = {"orderId": str(order_id)}
    try:
        data = await _async_call_api(settings.api_implus, "getMessagesByOrderId", payload)
    except Exception as e:
        LogHelper.log_warn(f"查询IMPlus消息失败 order_id={order_id}: {e}", {"orderId": str(order_id)})
        return []

    sessions = data.get("sessionList", [])
    messages = data.get("messages", [])
    if not sessions or not messages:
        return []

    # 取最近一个 session 的 gid 用于过滤消息
    sessions_sorted = sorted(sessions, key=lambda s: s.get("createTime", 0), reverse=True)
    latest_session = sessions_sorted[0]
    latest_gid = str(latest_session.get("gid", ""))

    # 7天时效过滤
    cutoff_ts = (datetime.now() - timedelta(days=settings.query_days)).timestamp() * 1000
    if latest_session.get("createTime", 0) < cutoff_ts:
        return []

    # 过滤：只取该 session 的文本消息 (msgtype == 0)
    text_messages = []
    for msg in messages:
        if msg.get("msgtype") != 0:
            continue
        from_jid = msg.get("fromJid", "")
        if f"{latest_gid}@" not in from_jid:
            continue
        if "/system" in from_jid:
            continue

        nick_name = msg.get("fromNickName", "")
        body = msg.get("messageBody", "")
        create_time = msg.get("createTime", 0)

        if not nick_name or not body:
            continue

        time_str = datetime.fromtimestamp(create_time / 1000).strftime("%Y-%m-%d %H:%M:%S") if create_time else ""
        text_messages.append({
            "发言人": nick_name,
            "内容": body[:200],
            "时间": time_str,
        })

    # 取最后 N 条
    max_msgs = settings.implus_max_messages
    return text_messages[-max_msgs:] if len(text_messages) > max_msgs else text_messages
