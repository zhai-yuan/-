"""
FastAPI 应用入口 - 路由、中间件、SOA 兼容
"""
import os
os.environ.setdefault("app_id", "100072638")
os.environ.setdefault("PAAS_APP_APPID", "100072638")

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

from app.models import IntentPredictionRequest, IntentPredictionResponse
from app.service import IntentPredictionService
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    yield


app = FastAPI(
    title="进线意图预测服务",
    description="基于历史会话摘要和用户受阻记录，预测客户下次进线意图",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = IntentPredictionService()


# ---- SOA 兼容: 强制解析 JSON (忽略 Content-Type) ----

async def _get_json_force(request: Request) -> dict[str, Any]:
    try:
        body_bytes = await request.body()
        if not body_bytes:
            return {}
        return json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return {}


# ---- 健康检查路由 ----

@app.get("/")
async def root():
    return {
        "service": "进线意图预测服务",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return "OK"


@app.get("/vi/health")
async def vi_health():
    return "OK"


@app.get("/checkhealth.json")
async def soa_health_check():
    return {"timestamp": time.time(), "ack": "success"}


@app.get("/stats")
async def stats():
    return prediction_service.get_stats()


@app.get("/_operationinfo")
async def operation_info():
    return [
        {
            "Name": "intentPrediction",
            "RequestMessage": {
                "orderId": "string",
                "sessionId": "string",
            },
        }
    ]


# ---- 核心预测路由 ----

async def _predict_core(req: IntentPredictionRequest) -> IntentPredictionResponse:
    tags = {"orderId": str(req.orderId), "sessionId": req.sessionId}
    LogHelper.log_info(f"收到预测请求: orderId={req.orderId}, sessionId={req.sessionId}", tags)
    result = await prediction_service.predict(
        order_id=req.orderId,
        session_id=req.sessionId,
        language=req.language,
    )
    LogHelper.log_info(f"预测完成: success={result.success}, 耗时={result.processingTime}s", tags)
    return result


@app.post("/intentPrediction", response_model=IntentPredictionResponse)
async def intent_prediction(req: IntentPredictionRequest):
    return await _predict_core(req)


async def _json_predict_from_request(request: Request) -> IntentPredictionResponse:
    data = await _get_json_force(request)
    try:
        req = IntentPredictionRequest(
            orderId=data.get("orderId"),
            sessionId=data.get("sessionId"),
            language=data.get("language"),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"请求参数错误: {str(e)}")
    return await _predict_core(req)


@app.post("/bjjson/intentPrediction", response_model=IntentPredictionResponse)
async def bjjson_intent_prediction(request: Request):
    return await _json_predict_from_request(request)


# 兼容路由：网关契约未登记/未完成改写时，直接命中后端
# 覆盖 /json 与 /bjjson 两种前缀 + camelCase 与全小写两种大小写
@app.post("/json/intentPrediction", response_model=IntentPredictionResponse)
@app.post("/json/intentprediction", response_model=IntentPredictionResponse)
@app.post("/bjjson/intentprediction", response_model=IntentPredictionResponse)
async def json_intent_prediction_compat(request: Request):
    return await _json_predict_from_request(request)


# ---- 批量评估触发（Zeus 定时调度 / 手动调试） ----

@app.post("/admin/triggerEvaluation")
async def trigger_evaluation(request: Request):
    """
    触发批量评估任务。

    Zeus 定时调度（不传参数）：
      - 执行前置检查（Hive 分区就绪等），失败立即返回错误码让 Zeus 重试
      - 前置检查通过后，提交后台异步执行，接口立即返回成功

    手动调试（传 limit 参数）：
      - 同步执行，等待完成后返回完整结果
    """
    if not settings.evaluation_manual_trigger_enabled:
        raise HTTPException(status_code=403, detail="手动触发未启用")

    # 从 query params 或 body 获取 date 和 limit
    date_param = request.query_params.get("date")
    limit_param = request.query_params.get("limit")

    if not date_param or not limit_param:
        try:
            body = await request.json()
            if not date_param:
                date_param = body.get("date")
            if not limit_param:
                limit_param = body.get("limit")
        except Exception:
            pass

    limit_val = int(limit_param) if limit_param else None

    from scheduler.batch_evaluation import run_batch_evaluation, preflight_check

    # 同步模式：传了 limit 用于手动调试，直接等待返回
    if limit_val:
        success, stats = await run_batch_evaluation(target_date=date_param, limit=limit_val)
        return {"success": success, "stats": stats}

    # 异步模式（Zeus 全量调度）：
    # Step 1 - 前置检查（秒级完成），失败则返回错误让 Zeus 触发重试
    check_ok, check_msg = await preflight_check(target_date=date_param)
    if not check_ok:
        LogHelper.log_warn(f"批量评估前置检查未通过: {check_msg}")
        return {"success": False, "message": check_msg}

    # Step 2 - 前置检查通过，提交后台执行
    asyncio.ensure_future(run_batch_evaluation(target_date=date_param, limit=None))
    LogHelper.log_info("批量评估任务已提交后台执行")
    return {"success": True, "message": "批量评估任务已提交后台执行"}
