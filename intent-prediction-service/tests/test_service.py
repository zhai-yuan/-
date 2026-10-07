"""测试核心编排逻辑"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.service import IntentPredictionService


@pytest.fixture
def service():
    return IntentPredictionService()


@pytest.fixture
def mock_summaries():
    """模拟从 SOA 获取的摘要数据"""
    return [
        {
            "type": "电话摘要",
            "callId": "call001",
            "summary": "客人咨询退票事宜，客服已告知退票条件",
            "summaryTime": "2026-03-27 15:42:28",
            "scene": "close",
            "operator": "system",
        },
        {
            "type": "电话摘要",
            "callId": "call002",
            "summary": "客人催促退款进度，客服承诺24小时内回复",
            "summaryTime": "2026-03-28 10:00:00",
            "scene": "close",
            "operator": "system",
        },
    ]


@pytest.fixture
def mock_llm_output():
    return json.dumps({
        "预测分析": "客人退票后一直未收到退款，上次客服承诺24小时回复，因此预计再次来电催促退款进度。",
        "预期提示一级": "退票",
        "预期提示二级": "催退款",
        "预期提示事件小结": "客户因退款未到账再次进线催促退款进度",
        "辅助话术": "您好，注意到您的退票申请正在处理中，请问是咨询退款进度吗？",
    }, ensure_ascii=False)


@pytest.mark.asyncio
async def test_predict_success(service, mock_summaries, mock_llm_output):
    """测试完整预测流程（mock 所有外部依赖）"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output):

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        mock_client.get_related_order_ids = AsyncMock(return_value=({123, 456}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[
            {"callID": "call001", "source": "电话"},
            {"callID": "call002", "source": "电话"},
        ])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_client.query_tel_summary = AsyncMock(side_effect=[
            [mock_summaries[0]],
            [mock_summaries[1]],
        ])
        mock_client.query_chat_summary = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="current_session")

        assert result.success is True
        assert result.data is not None
        assert result.data.level1Label == "退票"
        assert result.data.level2Label == "催退款"
        assert result.data.context.orderId == 123
        assert result.data.context.relatedOrderIds == [123, 456]
        assert result.data.context.ubtRecordCount == 0


@pytest.mark.asyncio
async def test_predict_passes_user_prompt_to_background_persist(
    service, mock_summaries, mock_llm_output
):
    """测试成功预测后后台落表任务会收到本轮 user prompt"""
    persist_mock = AsyncMock()
    service._persist_result = persist_mock

    with patch("app.service.settings") as mock_settings, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output):

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[
            {"callID": "call001", "source": "电话"},
        ])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_client.query_tel_summary = AsyncMock(return_value=[mock_summaries[0]])
        mock_client.query_chat_summary = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="current_session")
        await asyncio.sleep(0)

        assert result.success is True
        persist_mock.assert_called_once()
        persisted_result, user_prompt, confidence = persist_mock.call_args.args
        assert persisted_result is result.data
        assert "## 标签体系" in user_prompt
        assert "## 历史交互摘要" in user_prompt
        assert "客人咨询退票事宜" in user_prompt


@pytest.mark.asyncio
async def test_predict_no_summaries(service):
    """5 个主信号全为空时返回失败 (摘要/通知/投诉/IMPlus/TTS)"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts:

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.query_days = 7

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        assert result.success is False
        assert "无足够历史数据" in result.errorMessage


@pytest.mark.asyncio
async def test_predict_triggers_with_implus_only(service, mock_llm_output):
    """只有 IMPlus 会话消息（无摘要）也应触发 LLM 预测"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output) as mock_llm:

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[
            {"role": "user", "content": "我要退票"},
        ])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        mock_llm.assert_called_once()
        assert result.success is True
        assert result.data.context.implusMessageCount == 1
        assert result.data.context.summaryCount == 0


@pytest.mark.asyncio
async def test_predict_blocked_and_ubt_alone_do_not_trigger(service):
    """仅有受阻记录或浏览埋点（无 5 个主信号）时不触发预测"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm") as mock_llm:

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.query_days = 7

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[
            {"blockType": "退票受阻", "blockTime": "2026-05-27 10:00:00"},
        ])
        mock_client.get_incomecall_records = AsyncMock(return_value=[])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[
            {"page": "order_detail", "ts": "2026-05-27 10:00:00"},
        ])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        mock_llm.assert_not_called()
        assert result.success is False
        assert "无足够历史数据" in result.errorMessage


# ---- 去重逻辑测试 ----


@pytest.mark.asyncio
async def test_dedup_cache_hit_returns_history(service):
    """去重命中：DB 有记录时直接返回历史结果，不触发 LLM"""
    cached_row = {
        "orderId": 123,
        "relatedOrderIds": "[123, 456]",
        "sessionId": "s1",
        "predictionAnalysis": "历史分析",
        "level1Label": "退票",
        "level2Label": "催退款",
        "eventSummary": "历史小结",
        "suggestedScript": "历史话术",
    }

    with patch("app.service.settings") as mock_settings, \
         patch("app.service.get_db_helper") as mock_get_db, \
         patch("app.service.call_llm") as mock_llm:

        mock_settings.dedup_enabled = True
        mock_db = MagicMock()
        mock_db.query_prediction_by_session.return_value = cached_row
        mock_get_db.return_value = mock_db

        result = await service.predict(order_id=123, session_id="s1")

        assert result.success is True
        assert result.data.level1Label == "退票"
        assert result.data.level2Label == "催退款"
        assert result.data.predictionAnalysis == "历史分析"
        assert result.data.eventSummary == "历史小结"
        assert result.data.suggestedScript == "历史话术"
        assert result.data.context.relatedOrderIds == [123, 456]
        mock_llm.assert_not_called()
        assert service._cache_hit_count == 1


@pytest.mark.asyncio
async def test_dedup_cache_miss_triggers_prediction(service, mock_summaries, mock_llm_output):
    """去重未命中：DB 无记录时执行完整预测流程"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.get_db_helper") as mock_get_db, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output) as mock_llm:

        mock_settings.dedup_enabled = True
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        mock_db = MagicMock()
        mock_db.query_prediction_by_session.return_value = None
        mock_get_db.return_value = mock_db

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[
            {"callID": "call001", "source": "电话"},
        ])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_client.query_tel_summary = AsyncMock(return_value=[mock_summaries[0]])
        mock_client.query_chat_summary = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        mock_llm.assert_called_once()
        assert result.success is True
        assert result.data.level1Label == "退票"
        assert service._cache_hit_count == 0


@pytest.mark.asyncio
async def test_dedup_db_error_degrades_to_prediction(service, mock_summaries, mock_llm_output):
    """去重查询异常时降级为正常预测流程"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.get_db_helper") as mock_get_db, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output) as mock_llm:

        mock_settings.dedup_enabled = True
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        # DB 查询抛异常
        mock_db = MagicMock()
        mock_db.query_prediction_by_session.side_effect = RuntimeError("DB down")
        mock_get_db.return_value = mock_db

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[
            {"callID": "call001", "source": "电话"},
        ])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_client.query_tel_summary = AsyncMock(return_value=[mock_summaries[0]])
        mock_client.query_chat_summary = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        # DB 异常 → 降级到正常预测 → LLM 被调用
        mock_llm.assert_called_once()
        assert result.success is True
        assert service._cache_hit_count == 0


@pytest.mark.asyncio
async def test_dedup_disabled_skips_cache_check(service, mock_summaries, mock_llm_output):
    """dedup_enabled=False 时不查 DB，直接执行预测"""
    with patch("app.service.settings") as mock_settings, \
         patch("app.service.get_db_helper") as mock_get_db, \
         patch("app.service.summary_client") as mock_client, \
         patch("app.service.ubt_client") as mock_ubt, \
         patch("app.service.notification_client") as mock_notif, \
         patch("app.service.complaint_client") as mock_complaint, \
         patch("app.service.implus_client") as mock_implus, \
         patch("app.service.tts_client") as mock_tts, \
         patch("app.service.call_llm", return_value=mock_llm_output):

        mock_settings.dedup_enabled = False
        mock_settings.ubt_enabled = True
        mock_settings.notifications_enabled = True
        mock_settings.complaint_enabled = True
        mock_settings.implus_enabled = True
        mock_settings.tts_enabled = True
        mock_settings.order_status_enabled = True
        mock_settings.confidence_enabled = False
        mock_settings.query_days = 7

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        mock_client.get_related_order_ids = AsyncMock(return_value=({123}, {}))
        mock_client.query_user_blocked_records = AsyncMock(return_value=[])
        mock_client.get_incomecall_records = AsyncMock(return_value=[
            {"callID": "call001", "source": "电话"},
        ])
        mock_client.get_outcall_records = AsyncMock(return_value=[])
        mock_client.query_tel_summary = AsyncMock(return_value=[mock_summaries[0]])
        mock_client.query_chat_summary = AsyncMock(return_value=[])
        mock_ubt.query_order_detail_ubt = AsyncMock(return_value=[])
        mock_notif.get_notification_timeline = AsyncMock(return_value=[])
        mock_complaint.search_complaints = AsyncMock(return_value=[])
        mock_implus.get_implus_messages = AsyncMock(return_value=[])
        mock_tts.get_tts_outcall_results = AsyncMock(return_value=[])

        result = await service.predict(order_id=123, session_id="s1")

        # 不应调用 DB 查询
        mock_db.query_prediction_by_session.assert_not_called()
        assert result.success is True


@pytest.mark.asyncio
async def test_stats_includes_cache_hit_count(service):
    """stats 接口包含 cache_hit_count"""
    stats = service.get_stats()
    assert "cache_hit_count" in stats
    assert stats["cache_hit_count"] == 0
