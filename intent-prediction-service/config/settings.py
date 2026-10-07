"""
配置文件 - 进线意图预测服务
使用 QConfigHolder 从线上配置中心读取配置
所有配置项通过 @property 实现动态读取，线上修改后立即生效，无需重启服务
"""
import os

try:
    from bicommonkits.flight.bi.rm.common import QConfigHolder
    _HAS_QCONFIG = True
except ImportError:
    _HAS_QCONFIG = False


_CONFIG_MISSING = object()
_APP_ID = "100072638"


def _get_config_value(key: str, default=None, config_name: str = "base_config.json"):
    """从 QConfig 读取配置值，读不到则返回 default"""
    if not _HAS_QCONFIG:
        return default
    app_id = os.environ.get("app_id", _APP_ID)
    try:
        value = QConfigHolder.get_config_value(
            config_name, key, default=_CONFIG_MISSING, app_id=app_id,
        )
        if value is not _CONFIG_MISSING:
            return value

        if "." in key:
            root_key, *rest = key.split(".")
            root_val = QConfigHolder.get_config_value(
                config_name, root_key, default=_CONFIG_MISSING, app_id=app_id,
            )
            if root_val is not _CONFIG_MISSING and isinstance(root_val, dict):
                cur = root_val
                for part in rest:
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        return default
                return cur
    except Exception:
        pass

    return default


class Settings:
    """全局配置 - 所有字段通过 @property 动态读取 QConfig，线上修改后立即生效"""

    # ---- LLM 主模型配置 ----

    @property
    def llm_model(self):
        return _get_config_value("llm.model", "gemini-3.1-flash-lite-preview")

    @property
    def llm_base_url(self):
        return _get_config_value("llm.base_url", "http://aigw.fx.ctripcorp.com/llm/100000483")

    @property
    def llm_api_key(self):
        return _get_config_value("llm.api_key", "")

    @property
    def llm_max_retries(self):
        return _get_config_value("llm.max_retries", 2)

    # ---- LLM 兜底模型配置 ----

    @property
    def llm_fallback_model(self):
        return _get_config_value("llm_fallback.model", "Qwen3.5-27B")

    @property
    def llm_fallback_base_url(self):
        return _get_config_value("llm_fallback.base_url", "http://aigw.fx.ctripcorp.com/llm/100000483")

    @property
    def llm_fallback_api_key(self):
        return _get_config_value("llm_fallback.api_key", "")

    # ---- SOA API 地址 ----

    @property
    def api_order_detail(self):
        return _get_config_value("api.order_detail", "http://webapi.soa.ctripcorp.com/api/11817")

    @property
    def api_calling_center(self):
        return _get_config_value("api.calling_center", "http://webapi.soa.ctripcorp.com/api/14537")

    @property
    def api_tel_summary(self):
        return _get_config_value("api.tel_summary", "http://webapi.soa.ctripcorp.com/api/17307")

    @property
    def api_chat_summary(self):
        return _get_config_value("api.chat_summary", "http://webapi.soa.ctripcorp.com/api/15408")

    @property
    def api_user_behavior(self):
        return _get_config_value("api.user_behavior", "http://webapi.soa.ctripcorp.com/api/11504")

    @property
    def api_notification(self):
        return _get_config_value("api.notification", "http://webapi.soa.ctripcorp.com/api/15098")

    @property
    def api_wechat(self):
        return _get_config_value("api.wechat", "http://webapi.soa.ctripcorp.com/api/14464")

    @property
    def api_complaint(self):
        return _get_config_value("api.complaint", "http://webapi.soa.ctripcorp.com/api/14718")

    @property
    def api_implus(self):
        return _get_config_value("api.implus", "http://webapi.soa.ctripcorp.com/api/15408")

    @property
    def api_tts_outcall(self):
        return _get_config_value("api.tts_outcall", "http://webapi.soa.ctripcorp.com/api/14242")

    # ---- 超时配置 (秒) ----

    @property
    def api_timeout(self):
        return _get_config_value("timeout.api", 10)

    @property
    def llm_timeout(self):
        return _get_config_value("timeout.llm", 20)

    @property
    def total_timeout(self):
        return _get_config_value("timeout.total", 30)

    # ---- 数据查询 ----

    @property
    def query_days(self):
        return _get_config_value("query.days", 7)

    # ---- 并发控制 ----

    @property
    def max_concurrent_api_calls(self):
        return _get_config_value("concurrency.max_api_calls", 20)

    # ---- UBT 订单详情页埋点 (MyTrix SQL 引擎) ----

    @property
    def ubt_enabled(self) -> bool:
        """是否启用订单详情页 UBT 埋点数据。默认开启，可通过 QConfig 动态关闭。"""
        value = _get_config_value("ubt.enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def blocked_records_enabled(self) -> bool:
        """是否启用用户受阻记录查询。默认开启，可通过 QConfig 动态关闭。"""
        value = _get_config_value("features.blocked_records_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    # ---- Feature Flags (新数据源开关) ----

    @property
    def notifications_enabled(self) -> bool:
        value = _get_config_value("features.notifications_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def complaint_enabled(self) -> bool:
        value = _get_config_value("features.complaint_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def order_status_enabled(self) -> bool:
        value = _get_config_value("features.order_status_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def implus_enabled(self) -> bool:
        value = _get_config_value("features.implus_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def tts_enabled(self) -> bool:
        value = _get_config_value("features.tts_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def dedup_enabled(self) -> bool:
        """是否启用同 session 去重（首次预测结果命中直接返回）。默认开启。"""
        value = _get_config_value("features.dedup_enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    # ---- 置信度过滤配置 ----

    @property
    def confidence_threshold(self) -> int:
        """置信度阈值，低于此值的预测不下发给客服。默认40。"""
        return _get_config_value("confidence.threshold", 40)

    @property
    def confidence_enabled(self) -> bool:
        """是否启用置信度过滤。默认开启。"""
        value = _get_config_value("confidence.enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    # ---- 通知限制参数 ----

    @property
    def notification_max_count(self) -> int:
        return _get_config_value("notification.max_count", 10)

    @property
    def notification_content_max_chars(self) -> int:
        return _get_config_value("notification.content_max_chars", 300)

    @property
    def implus_max_messages(self) -> int:
        return _get_config_value("implus.max_messages", 15)

    @property
    def tts_max_count(self) -> int:
        return _get_config_value("tts.max_count", 5)

    @property
    def ubt_mytrix_url(self):
        return _get_config_value(
            "ubt.mytrix_url",
            "http://mytrixhub.flight.ctripcorp.com/mytrix/meta2/sql_engine/exec",
        )

    @property
    def ubt_employee_id(self):
        return _get_config_value("ubt.employee_id", "TR052031")

    @property
    def ubt_recent_count(self):
        return _get_config_value("ubt.recent_count", 3)

    # ---- MySQL 配置 ----

    @property
    def mysql_dal_cluster_name(self):
        return _get_config_value("mysql.dal_cluster_name", "fltbidataservicedb_dalcluster")

    @property
    def mysql_app_id(self):
        return _get_config_value("mysql.app_id", _APP_ID)

    # ---- 批量评估任务配置 ----

    @property
    def evaluation_enabled(self) -> bool:
        value = _get_config_value("evaluation.enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def evaluation_concurrency(self) -> int:
        return _get_config_value("evaluation.concurrency", 5)

    @property
    def evaluation_batch_size(self) -> int:
        return _get_config_value("evaluation.batch_size", 500)

    @property
    def evaluation_hive_table(self) -> str:
        return _get_config_value(
            "evaluation.hive_table",
            "dwflt.cdm_service_cdr_contact_msg_di",
        )

    @property
    def evaluation_llm_model(self) -> str:
        return _get_config_value("evaluation.llm_model", "Qwen3.5-27B-FP8")

    @property
    def evaluation_llm_base_url(self) -> str:
        return _get_config_value(
            "evaluation.llm_base_url",
            "http://port8033.ocp21pronwrocwjw-tr034784-0-svc.gps-ali-pro1.cloud.ctripcorp.com/v1",
        )

    @property
    def evaluation_llm_api_key(self) -> str:
        return _get_config_value("evaluation.llm_api_key", "EMPTY")

    @property
    def evaluation_manual_trigger_enabled(self) -> bool:
        value = _get_config_value("evaluation.manual_trigger_enabled", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @property
    def evaluation_mcp_url(self) -> str:
        return _get_config_value(
            "evaluation.mcp_url",
            "http://adhoc-query-mcp-function.faas.ctripcorp.com/mcp",
        )

    @property
    def evaluation_mcp_token(self) -> str:
        return _get_config_value(
            "evaluation.mcp_token",
            "ada_1e91b97cb7e65b0d563c0508b21fea6d6fc79094e5fac45a4388c7e6ed071197",
        )


settings = Settings()
