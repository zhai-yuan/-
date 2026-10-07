"""测试 TTS 外呼客户端"""
import json
import pytest
from unittest.mock import AsyncMock, patch

from clients.tts_client import (
    _extract_tts_content,
    _normalize_tts_result,
    get_tts_outcall_results,
)
from config.prompts import format_tts_records


def test_extract_tts_content_normal():
    """正常嵌套JSON中提取TTS文本"""
    inner = json.dumps({
        "IvrFlow": "200-002",
        "TTS1": "5月31日，HO1093，郑州至巴彦淖尔的航班，接航司通知，航班取消。",
        "TTS2": "",
        "TTS3": "如需申请退票或改签，您可以在携程客户端中自助操作。",
    })
    outer = json.dumps({"content": inner, "recordGUID": "123"})
    detail = {"Content": outer}
    result = _extract_tts_content(detail)
    assert "航班取消" in result
    assert "如需申请退票或改签" in result


def test_extract_tts_content_empty():
    """Content为空时返回空字符串"""
    assert _extract_tts_content({}) == ""
    assert _extract_tts_content({"Content": ""}) == ""


def test_extract_tts_content_malformed_json():
    """JSON格式异常时不抛异常"""
    assert _extract_tts_content({"Content": "not json"}) == ""


def test_normalize_tts_result_success():
    """标准化成功的外呼结果"""
    inner = json.dumps({
        "TTS1": "航班取消通知",
        "TTS2": "",
        "TTS3": None,
    })
    outer = json.dumps({"content": inner})
    item = {
        "FlightOutCallTaskId": 515596040,
        "BusinessId": 1128099418839857,
        "OrderId": 1128147995614961,
        "BusinessType": 1,
        "CreateTime": "2026-05-20 13:44:03.0",
        "DetailOutCallResult": {
            "Content": outer,
            "CallResult": "A",
        },
    }
    result = _normalize_tts_result(item)
    assert result is not None
    assert result["content"] == "航班取消通知"
    assert result["callResult"] == "已接听"
    assert result["businessType"] == "改签"
    assert result["time"] == "2026-05-20 13:44:03.0"


def test_normalize_tts_result_no_content():
    """无TTS内容时返回None"""
    item = {
        "BusinessType": 1,
        "CreateTime": "2026-05-20 13:44:03.0",
        "DetailOutCallResult": {"Content": "", "CallResult": "A"},
    }
    assert _normalize_tts_result(item) is None


def test_normalize_tts_result_unanswered():
    """未接听标记"""
    inner = json.dumps({"TTS1": "通知内容", "TTS2": "", "TTS3": None})
    outer = json.dumps({"content": inner})
    item = {
        "BusinessType": 0,
        "CreateTime": "2026-05-20 14:00:00.0",
        "DetailOutCallResult": {"Content": outer, "CallResult": "B"},
    }
    result = _normalize_tts_result(item)
    assert result["callResult"] == "未接听"
    assert result["businessType"] == "退票"


@pytest.mark.asyncio
async def test_get_tts_outcall_results_success():
    """正常返回外呼结果列表"""
    inner = json.dumps({"TTS1": "航班取消", "TTS2": "", "TTS3": ""})
    outer = json.dumps({"content": inner})
    mock_response = {
        "OutCallResultList": [
            {
                "BusinessType": 1,
                "CreateTime": "2026-05-20 13:44:03.0",
                "DetailOutCallResult": {"Content": outer, "CallResult": "A"},
            }
        ]
    }
    with patch("clients.tts_client._async_call_api", new_callable=AsyncMock, return_value=mock_response):
        results = await get_tts_outcall_results(123)
        assert len(results) == 1
        assert results[0]["content"] == "航班取消"


@pytest.mark.asyncio
async def test_get_tts_outcall_results_api_failure():
    """API调用失败返回空列表"""
    with patch("clients.tts_client._async_call_api", new_callable=AsyncMock, side_effect=Exception("timeout")):
        results = await get_tts_outcall_results(123)
        assert results == []


def test_format_tts_records_empty():
    """空列表返回空字符串"""
    assert format_tts_records([]) == ""


def test_format_tts_records_normal():
    """正常格式化TTS记录"""
    records = [
        {
            "time": "2026-05-20 13:44:03.0",
            "content": "航班取消通知",
            "callResult": "已接听",
            "businessType": "改签",
        },
        {
            "time": "2026-05-20 14:31:02.0",
            "content": "航班调整通知",
            "callResult": "未接听",
            "businessType": "退票",
        },
    ]
    result = format_tts_records(records)
    assert "【外呼1】" in result
    assert "【外呼2】" in result
    assert "航班取消通知" in result
    assert "已接听" in result
    assert "改签" in result
