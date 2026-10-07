"""测试 prompt 拼装和标签工具函数"""
import json
from pathlib import Path
from config.prompts import (
    build_label_reference,
    build_user_prompt,
    format_summaries,
    format_blocked_records,
)


def _load_label_mapping() -> dict:
    path = Path(__file__).parent.parent / "data" / "label_mapping.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_build_label_reference():
    mapping = {"退票": ["催退款", "催退票"], "改签": ["催改签"]}
    ref = build_label_reference(mapping)
    assert "【退票】催退款、催退票" in ref
    assert "【改签】催改签" in ref


def test_build_label_reference_full():
    mapping = _load_label_mapping()
    ref = build_label_reference(mapping)
    assert "【出票】" in ref
    assert "【增值服务】" in ref
    assert "【退票】" in ref
    assert "【预订】" in ref


def test_format_summaries_empty():
    assert format_summaries([]) == "无历史交互摘要"


def test_format_summaries_normal():
    summaries = [
        {"type": "电话摘要", "summaryTime": "2026-03-27 15:42:28", "summary": "客人申请因病退票被拒"},
        {"type": "会话摘要", "summaryTime": "2026-03-27 16:00:00", "summary": "客人咨询退票进度"},
    ]
    result = format_summaries(summaries)
    assert "【摘要1】" in result
    assert "电话摘要" in result
    assert "【摘要2】" in result
    assert "会话摘要" in result


def test_format_blocked_records_empty():
    assert format_blocked_records([]) == "无"


def test_format_blocked_records_normal():
    records = [
        {"createTime": "2026-03-27 10:00:00", "pageCode": "orderDetail", "blockedErrorMessage": "退票按钮不可用"},
    ]
    result = format_blocked_records(records)
    assert "【受阻1】" in result
    assert "退票按钮不可用" in result


def test_build_user_prompt_contains_all_sections():
    prompt = build_user_prompt("标签内容", "摘要内容", "受阻内容", "浏览内容")
    assert "## 标签体系" in prompt
    assert "标签内容" in prompt
    assert "## 历史交互摘要" in prompt
    assert "摘要内容" in prompt
    assert "## 用户受阻记录" in prompt
    assert "受阻内容" in prompt
    assert "## 订单详情页浏览记录" in prompt
    assert "浏览内容" in prompt
    assert "请预测客户下次进线意图" in prompt
