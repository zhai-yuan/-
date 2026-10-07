"""MySQL 落表工具 - 将意图预测结果写入 intent_prediction_results 表"""
import json
import os
from typing import Any

try:
    from pymysql.converters import escape_string
    from bicommonkits.flight.bi.rm.common.db import MysqlPoolDB
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    def escape_string(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "\\'")
    MysqlPoolDB = None  # type: ignore

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
from app.models import PredictionResult


_TABLE_NAME = "intent_prediction_results"
_COLS = [
    "orderId",
    "relatedOrderIds",
    "sessionId",
    "userPrompt",
    "predictionAnalysis",
    "level1Label",
    "level2Label",
    "eventSummary",
    "suggestedScript",
    "confidence",
]


def _escape(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    return f"'{escape_string(str(val))}'"


class DbHelper:
    def __init__(self):
        if not _DB_AVAILABLE:
            raise RuntimeError("MysqlPoolDB 不可用, 本地环境未安装 bicommonkits/pymysql")
        # 设置 PAAS_APP_APPID 环境变量（MysqlPoolDB 通过 DAL 方式连接时需要）
        os.environ.setdefault("PAAS_APP_APPID", settings.mysql_app_id)
        self.mysql_pool = MysqlPoolDB(
            dal_cluster_name=settings.mysql_dal_cluster_name,
        )

    def insert_prediction_result(self, result: PredictionResult, user_prompt: str, confidence: int = 0) -> bool:
        """
        插入意图预测结果到 intent_prediction_results 表。
        使用 INSERT IGNORE：sessionId 唯一键冲突时忽略，保留首次预测结果不被覆盖。
        """
        ctx = result.context
        row = [
            ctx.orderId,
            json.dumps(ctx.relatedOrderIds, ensure_ascii=False),
            ctx.sessionId,
            user_prompt,
            result.predictionAnalysis,
            result.level1Label,
            result.level2Label,
            result.eventSummary,
            result.suggestedScript,
            confidence,
        ]

        col_str = ",".join(f"`{c}`" for c in _COLS)
        values_clause = "(" + ",".join(_escape(v) for v in row) + ")"

        sql = f"INSERT IGNORE INTO {_TABLE_NAME} ({col_str}) VALUES {values_clause};"

        try:
            self.mysql_pool.execute(sql)
            LogHelper.log_info(
                f"预测结果落表成功: sessionId={ctx.sessionId}, orderId={ctx.orderId}",
                {"orderId": str(ctx.orderId), "sessionId": ctx.sessionId},
            )
            return True
        except Exception as e:
            LogHelper.log_error(
                f"预测结果落表失败: sessionId={ctx.sessionId}, error={str(e)}",
                {"orderId": str(ctx.orderId), "sessionId": ctx.sessionId},
            )
            return False

    def query_prediction_by_session(self, session_id: str) -> dict | None:
        """
        根据 sessionId 查询已有的预测结果（命中 idx_sessionId 唯一索引）。
        用于去重：同一 session 不重复预测。
        返回 dict（包含核心预测字段）或 None（无记录）。
        查询失败时返回 None，降级为正常预测流程。
        """
        sql = (
            f"SELECT orderId, relatedOrderIds, sessionId, "
            f"predictionAnalysis, level1Label, level2Label, "
            f"eventSummary, suggestedScript "
            f"FROM {_TABLE_NAME} "
            f"WHERE sessionId = {_escape(session_id)} "
            f"LIMIT 1"
        )
        try:
            rows = self.mysql_pool.select(sql)
            if rows:
                return rows[0]
            return None
        except Exception as e:
            LogHelper.log_error(
                f"去重查询失败: sessionId={session_id}, error={str(e)}",
                {"sessionId": session_id},
            )
            return None

    def query_predictions_by_date(self, date_str: str) -> list[dict]:
        """
        查询指定日期的未评估预测记录。
        date_str 格式: YYYY-MM-DD
        返回 list[dict]，每个 dict 包含: sessionId, orderId, level1Label, level2Label,
        eventSummary, predictionAnalysis, suggestedScript, userPrompt
        """
        sql = (
            f"SELECT sessionId, orderId, level1Label, level2Label, "
            f"eventSummary, predictionAnalysis, suggestedScript, userPrompt "
            f"FROM {_TABLE_NAME} "
            f"WHERE DATE(DataChange_CreateTime) = '{escape_string(date_str)}' "
            f"AND isAccurate IS NULL"
        )
        try:
            rows = self.mysql_pool.select(sql)
            if not rows:
                return []
            LogHelper.log_info(f"查询到 {len(rows)} 条未评估的预测记录 (date={date_str})")
            return rows
        except Exception as e:
            LogHelper.log_error(f"查询预测记录失败 date={date_str}: {str(e)}")
            return []

    def update_evaluation_result(
        self,
        session_id: str,
        is_accurate: bool,
        call_summary: str,
        is_predictable: int | None = None,
        supporting_signals: str = "",
        missing_info: str = "",
        is_script_direction_accurate: bool | None = None,
    ) -> bool:
        """
        更新单条预测记录的评估结果（含可预测性分析和话术方向评估）。
        """
        accurate_val = 1 if is_accurate else 0
        predictable_clause = (
            f"isPredictable = {is_predictable}" if is_predictable is not None else "isPredictable = NULL"
        )
        script_direction_clause = (
            f"isScriptDirectionAccurate = {1 if is_script_direction_accurate else 0}"
            if is_script_direction_accurate is not None
            else "isScriptDirectionAccurate = NULL"
        )
        sql = (
            f"UPDATE {_TABLE_NAME} "
            f"SET isAccurate = {accurate_val}, "
            f"callSummary = {_escape(call_summary)}, "
            f"{predictable_clause}, "
            f"supportingSignals = {_escape(supporting_signals or None)}, "
            f"missingInfo = {_escape(missing_info or None)}, "
            f"{script_direction_clause}, "
            f"dataSource = 'C' "
            f"WHERE sessionId = {_escape(session_id)}"
        )
        try:
            self.mysql_pool.execute(sql)
            return True
        except Exception as e:
            LogHelper.log_error(
                f"评估结果更新失败: sessionId={session_id}, error={str(e)}"
            )
            return False


_db_helper: DbHelper | None = None


def get_db_helper() -> DbHelper:
    global _db_helper
    if _db_helper is None:
        _db_helper = DbHelper()
    return _db_helper
