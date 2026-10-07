"""测试订单详情页埋点客户端"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clients import ubt_client
from clients.ubt_client import (
    _shrink,
    _extract_clicked_modules,
    _dedup_and_limit,
    query_order_detail_ubt,
)
from config.prompts import format_ubt_records


FIXTURE = Path(__file__).parent / "fixtures" / "ubt_sample.json"


def _load_fixture() -> dict:
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_extract_clicked_modules_filters_by_click_flag():
    raw = json.dumps([
        {"show": 1, "title": "回退", "module": "头部", "type": "", "click": 0},
        {"show": 1, "title": "2001", "module": "Card_Bottom", "type": "Jump_Checkin", "click": 1},
        {"show": 1, "title": "RefundReBookPolicy", "module": "Card_Flight", "type": "Interactive", "click": 1},
    ])
    result = _extract_clicked_modules(raw)
    assert result == ["Card_Bottom/2001_Jump_Checkin", "Card_Flight/RefundReBookPolicy_Interactive"]


def test_extract_clicked_modules_handles_malformed():
    assert _extract_clicked_modules("not-json") == []
    assert _extract_clicked_modules(None) == []
    assert _extract_clicked_modules("{}") == []


def test_shrink_extracts_four_fields():
    sample = _load_fixture()["data"][0]
    out = _shrink(sample)
    assert set(out.keys()) == {"viewed_at", "trigger", "stay_ms", "clicked_modules"}
    assert out["viewed_at"] == "2026-05-08 12:07:54.687"
    assert out["trigger"] == "exit"
    assert out["stay_ms"] == 9639
    assert out["clicked_modules"] == ["Card_Bottom/2001_Jump_Checkin"]


def test_shrink_handles_malformed_data_info():
    # data_info 非 JSON
    assert _shrink({"start_time": "t", "data_info": "{invalid"})["clicked_modules"] == []
    # data_info 缺字段
    result = _shrink({"start_time": "t", "data_info": "{}"})
    assert result["trigger"] == ""
    assert result["stay_ms"] == 0
    # 整行缺字段
    result = _shrink({})
    assert result["viewed_at"] == ""
    assert result["stay_ms"] == 0


def test_dedup_and_limit_keeps_latest_per_pvid_and_caps():
    rows = _load_fixture()["data"]
    # fixture 中 pvid 5740 有 2 条，应只保留 start_time 最大的那条（12:07:54.687）
    top3 = _dedup_and_limit(rows, limit=3)
    assert len(top3) == 3
    pvids = [r["pvid"] for r in top3]
    # 5 个不同 pvid：5740/5745/5751/5760/5770，按 start_time DESC 应取后三个
    assert pvids == [5770, 5760, 5751]


def test_dedup_and_limit_pvid_5740_keeps_latest():
    rows = _load_fixture()["data"]
    all_deduped = _dedup_and_limit(rows, limit=100)
    by_pvid = {r["pvid"]: r for r in all_deduped}
    assert by_pvid[5740]["start_time"] == "2026-05-08 12:07:54.687"


@pytest.mark.asyncio
async def test_query_order_detail_ubt_end_to_end():
    """模拟 MyTrix 返回 fixture 数据，验证端到端去重+瘦身+限量。"""
    fixture = _load_fixture()
    with patch.object(ubt_client, "_call_mytrix", return_value=fixture):
        result = await query_order_detail_ubt(order_id=1128147843141834, limit=3)

    assert len(result) == 3
    # 最近一条应为 14:00:00（pvid 5770），内容结构正确
    assert result[0]["viewed_at"] == "2026-05-08 14:00:00.000"
    assert all(set(r.keys()) == {"viewed_at", "trigger", "stay_ms", "clicked_modules"} for r in result)
    # 最早入选的 pvid 5751 应有点击模块
    assert result[2]["clicked_modules"] == ["Card_Flight/RefundReBookPolicy_Interactive"]


@pytest.mark.asyncio
async def test_query_order_detail_ubt_failure_returns_empty():
    """MyTrix 抛异常时返空列表，不抛。"""
    with patch.object(ubt_client, "_call_mytrix", side_effect=RuntimeError("boom")):
        result = await query_order_detail_ubt(order_id=123, limit=3)
    assert result == []


@pytest.mark.asyncio
async def test_query_order_detail_ubt_empty_response():
    """上游返回空或结构异常时返空列表。"""
    with patch.object(ubt_client, "_call_mytrix", return_value={"data": []}):
        assert await query_order_detail_ubt(order_id=123, limit=3) == []
    with patch.object(ubt_client, "_call_mytrix", return_value={"other": "x"}):
        assert await query_order_detail_ubt(order_id=123, limit=3) == []


def test_format_ubt_records_empty():
    assert format_ubt_records([]) == "无"


def test_format_ubt_records_renders_key_signals():
    records = [
        {"viewed_at": "2026-05-08 12:07:54", "trigger": "exit", "stay_ms": 9639,
         "clicked_modules": ["Card_Bottom/立即值机"]},
        {"viewed_at": "2026-05-08 12:08:04", "trigger": "background", "stay_ms": 19789,
         "clicked_modules": []},
    ]
    out = format_ubt_records(records)
    assert "【浏览1】" in out
    assert "停留：9.6s" in out
    assert "离开方式：exit" in out
    assert "Card_Bottom/立即值机" in out
    assert "【浏览2】" in out
    assert "无点击" in out
