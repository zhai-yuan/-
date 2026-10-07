"""测试 FastAPI 路由"""
import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.models import IntentPredictionResponse, PredictionResult, PredictionContext


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert "version" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == "OK"


def test_vi_health(client):
    resp = client.get("/vi/health")
    assert resp.status_code == 200


def test_checkhealth_json(client):
    resp = client.get("/checkhealth.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert data["ack"] == "success"


def test_operationinfo(client):
    resp = client.get("/_operationinfo")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["Name"] == "intentPrediction"


def _mock_success_response():
    return IntentPredictionResponse(
        success=True,
        data=PredictionResult(
            predictionAnalysis="分析",
            level1Label="退票",
            level2Label="催退票",
            eventSummary="小结",
            suggestedScript="话术",
            context=PredictionContext(
                orderId=123,
                relatedOrderIds=[123],
                sessionId="s1",
                summaryCount=2,
                blockedRecordCount=0,
                queryTimeRange={"start": "2026-04-08T00:00:00", "end": "2026-04-15T23:59:59"},
            ),
        ),
        processingTime=1.5,
    )


def test_intent_prediction_route(client):
    with patch("app.main.prediction_service") as mock_svc:
        mock_svc.predict = AsyncMock(return_value=_mock_success_response())
        resp = client.post(
            "/intentPrediction",
            json={"orderId": 123, "sessionId": "s1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["level1Label"] == "退票"


def test_bjjson_intent_prediction_route(client):
    with patch("app.main.prediction_service") as mock_svc:
        mock_svc.predict = AsyncMock(return_value=_mock_success_response())
        resp = client.post(
            "/bjjson/intentPrediction",
            content=json.dumps({"orderId": 123, "sessionId": "s1"}),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


def test_intent_prediction_missing_params(client):
    resp = client.post(
        "/intentPrediction",
        json={"orderId": 123},
    )
    assert resp.status_code == 422
