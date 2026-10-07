"""
调度器模块 - 批量评估定时任务

定时触发已迁移至 Zeus 平台（cron: 0 0 5 * * ?），通过 HTTP 调用
/admin/triggerEvaluation 接口实现。本模块仅保留生命周期钩子的兼容接口。
"""

try:
    from bicommonkits.flight.bi.rm.common import LogHelper
except ImportError:
    import logging
    class LogHelper:
        _logger = logging.getLogger("intent-prediction")
        @staticmethod
        def log_info(msg, tag=None): LogHelper._logger.info(msg)


def setup_scheduler():
    """兼容接口 - 定时调度已迁移至 Zeus 平台"""
    LogHelper.log_info("批量评估定时任务由 Zeus 平台调度，本地调度器已禁用")


def shutdown_scheduler():
    """兼容接口 - 无需清理"""
    pass
