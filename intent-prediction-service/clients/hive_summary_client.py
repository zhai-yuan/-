"""
Hive 会话数据客户端
通过 bigdata-adhoc-mcp 服务的 bigdataQuery 工具查询 Hive 表，
获取原始进线会话内容（manual_content）用于批量评估。
"""
import json
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


def _call_mcp_bigdata_query(sql: str, limit_disable: bool = False) -> dict:
    """
    通过 MCP 协议调用 bigdataQuery 工具执行 SQL。
    返回格式: {"success": True, "columns": [...], "rows": [[...], ...], "total_rows": N}
    失败时: {"success": False, "msg": "..."}
    """
    # MCP JSON-RPC 请求格式
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": "bigdataQuery",
            "arguments": {
                "sql": sql,
                "engine": "STARROCKS_HIVE",
            },
        },
    }
    if limit_disable:
        payload["params"]["arguments"]["limit_disable"] = "true"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-bbzai-mcp-token": settings.evaluation_mcp_token,
    }

    resp = requests.post(
        settings.evaluation_mcp_url,
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()

    # MCP Streamable HTTP 返回 SSE 格式: "event: message\ndata: {...}\n\n"
    # 注意: Content-Type 为 text/event-stream 未指定 charset，
    # requests 会默认按 ISO-8859-1 解码导致中文乱码，需显式用 UTF-8 解码
    rpc_resp = _parse_sse_response(resp.content.decode("utf-8"))

    if "error" in rpc_resp:
        return {"success": False, "msg": rpc_resp["error"].get("message", str(rpc_resp["error"]))}

    result = rpc_resp.get("result", {})
    content_list = result.get("content", [])
    if not content_list:
        return {"success": False, "msg": "MCP 返回内容为空"}

    # 解析第一个 text content
    text_content = content_list[0].get("text", "")
    try:
        data = json.loads(text_content)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "msg": f"MCP 返回内容解析失败: {text_content[:200]}"}

    return data


def _parse_sse_response(text: str) -> dict:
    """解析 SSE (Server-Sent Events) 格式的响应，提取 JSON-RPC 结果"""
    for line in text.split("\n"):
        if line.startswith("data: "):
            json_str = line[6:]  # 去掉 "data: " 前缀
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    # 如果没有 SSE 格式，尝试直接解析整体为 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": {"message": f"无法解析 MCP 响应: {text[:200]}"}}


def _rows_to_dicts(columns: list[str], rows: list[list]) -> list[dict]:
    """将列式格式 (columns + rows) 转为 list[dict]"""
    return [dict(zip(columns, row)) for row in rows]


def check_partition_ready(date_str: str) -> bool:
    """
    检查 Hive 分区是否存在数据。
    date_str 格式: YYYY-MM-DD
    返回 True 表示分区有数据，False 表示尚未就绪。
    """
    table = settings.evaluation_hive_table
    sql = f"SELECT 1 AS flag FROM {table} WHERE d = '{date_str}' LIMIT 1"
    try:
        resp = _call_mcp_bigdata_query(sql)
        if not resp.get("success"):
            LogHelper.log_warn(f"检查 Hive 分区就绪失败: {resp.get('msg', '')}")
            return False
        rows = resp.get("rows", [])
        return len(rows) > 0
    except Exception as e:
        LogHelper.log_warn(f"检查 Hive 分区就绪状态失败 d={date_str}: {e}")
        return False


def _format_manual_content(raw: str) -> str:
    """
    将原始 contact_manual_msg_list 转为可读对话格式。
    原始格式: 2026-05-27 12:36:21#S#内容###2026-05-27 12:40:03#U#内容###...
    输出格式:
      [12:36:21] 客服：内容
      [12:40:03] 用户：内容
    """
    if not raw:
        return ""
    # 按 ### 分割消息
    segments = raw.split("###")
    lines = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 格式: timestamp#role#content
        parts = seg.split("#", 2)
        if len(parts) < 3:
            # 兜底：旧格式 S#content 或 U#content
            if seg.startswith("S#"):
                lines.append(f"客服：{seg[2:]}")
            elif seg.startswith("U#"):
                lines.append(f"用户：{seg[2:]}")
            else:
                lines.append(seg)
            continue

        timestamp, role, content = parts[0], parts[1], parts[2]
        # 提取时间中的 HH:MM:SS 部分
        time_part = timestamp.strip().split(" ")[-1] if " " in timestamp else timestamp.strip()
        if role == "S":
            lines.append(f"[{time_part}] 客服：{content}")
        elif role == "U":
            lines.append(f"[{time_part}] 用户：{content}")
        else:
            lines.append(f"[{time_part}] {content}")
    return "\n".join(lines)


def query_conversations_by_session_ids(date_str: str, session_ids: list[str]) -> dict[str, str]:
    """
    按 sessionId (contact_id) 列表分批从 Hive 查询原始会话内容。
    返回 {session_id: formatted_conversation} 的字典。
    date_str 格式: YYYY-MM-DD
    """
    if not session_ids:
        return {}

    table = settings.evaluation_hive_table
    batch_size = settings.evaluation_batch_size
    result: dict[str, str] = {}

    for i in range(0, len(session_ids), batch_size):
        batch = session_ids[i:i + batch_size]
        ids_str = ",".join(f"'{sid}'" for sid in batch)
        sql = (
            f"SELECT contact_id, contact_manual_msg_list "
            f"FROM {table} "
            f"WHERE d = '{date_str}' "
            f"AND contact_id IN ({ids_str}) "
            f"LIMIT {len(batch)}"
        )
        try:
            resp = _call_mcp_bigdata_query(sql, limit_disable=True)
            if not resp.get("success"):
                LogHelper.log_warn(
                    f"查询 Hive 会话数据失败 batch={i//batch_size + 1}: {resp.get('msg', '')}"
                )
                continue

            columns = resp.get("columns", [])
            rows = resp.get("rows", [])
            if not columns or not rows:
                continue

            # 找到列索引
            try:
                contact_id_idx = columns.index("contact_id")
                content_idx = columns.index("contact_manual_msg_list")
            except ValueError:
                LogHelper.log_warn(f"Hive 返回列中缺少 contact_id 或 contact_manual_msg_list: {columns}")
                continue

            for row in rows:
                contact_id = str(row[contact_id_idx]) if row[contact_id_idx] else ""
                raw_content = str(row[content_idx]) if row[content_idx] else ""

                if contact_id and raw_content:
                    formatted = _format_manual_content(raw_content)
                    if formatted:
                        result[contact_id] = formatted

        except Exception as e:
            LogHelper.log_warn(
                f"查询 Hive 会话数据失败 batch={i//batch_size + 1}: {e}"
            )
            continue

    LogHelper.log_info(
        f"Hive 会话查询完成: 请求 {len(session_ids)} 条, 匹配到 {len(result)} 条"
    )
    return result
