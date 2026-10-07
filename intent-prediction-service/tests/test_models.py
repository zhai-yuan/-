"""测试 Pydantic 请求/响应模型"""
import pytest
from app.models import (
    IntentPredictionRequest,
    IntentPredictionResponse,
    PredictionResult,
    PredictionContext,
)


def test_request_valid():
    req = IntentPredictionRequest(orderId=1128146287853336, sessionId="abc123")
    assert req.orderId == 1128146287853336
    assert req.sessionId == "abc123"


def test_request_missing_order_id():
    with pytest.raises(Exception):
        IntentPredictionRequest(sessionId="abc123")


def test_request_missing_session_id():
    with pytest.raises(Exception):
        IntentPredictionRequest(orderId=123)


def test_response_success():
    ctx = PredictionContext(
        orderId=123,
        relatedOrderIds=[123, 456],
        sessionId="s1",
        summaryCount=5,
        blockedRecordCount=2,
        queryTimeRange={"start": "2026-04-08T00:00:00", "end": "2026-04-15T23:59:59"},
    )
    result = PredictionResult(
        predictionAnalysis="分析内容",
        level1Label="退票",
        level2Label="催退票",
        eventSummary="客户因退票进度缓慢再次来电催促",
        suggestedScript="您好，注意到您的退票申请正在处理中，请问是咨询退票进度吗？",
        context=ctx,
    )
    resp = IntentPredictionResponse(
        success=True,
        data=result,
        errorMessage=None,
        processingTime=2.5,
    )
    assert resp.success is True
    assert resp.data.level1Label == "退票"
    assert resp.processingTime == 2.5


def test_response_failure():
    resp = IntentPredictionResponse(
        success=False,
        data=None,
        errorMessage="无足够历史数据进行预测",
        processingTime=0.3,
    )
    assert resp.success is False
    assert resp.data is None
    assert resp.errorMessage == "无足够历史数据进行预测"
