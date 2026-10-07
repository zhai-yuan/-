#!/usr/bin/env python3
"""开发环境启动脚本"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

try:
    from bicommonkits.flight.bi.rm.common import LogHelper
except ImportError:
    import logging
    class LogHelper:
        _logger = logging.getLogger("intent-prediction")
        @staticmethod
        def log_info(msg, tag=None): LogHelper._logger.info(msg)


def main():
    Path("logs").mkdir(exist_ok=True)

    LogHelper.log_info("启动 进线意图预测服务 v1.0.0")
    LogHelper.log_info("服务地址: http://0.0.0.0:8000")
    LogHelper.log_info("API文档: http://0.0.0.0:8000/docs")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
