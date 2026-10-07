"""测试 LLM 输出解析和标签校验"""
import json
from clients.llm_client import parse_prediction


SAMPLE_LABEL_MAPPING = {
    "退票": ["催退款", "催退票", "咨询退票条件及办理"],
    "改签": ["催改签", "咨询改签条件及办理"],
    "求助、投诉": ["退票费、改签费高", "服务态度、技巧"],
}


def test_parse_clean_json():
    raw = json.dumps({
        "预测分析": "分析",
        "预期提示一级": "退票",
        "预期提示二级": "催退票",
        "预期提示事件小结": "小结",
        "辅助话术": "话术",
    }, ensure_ascii=False)
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert result["预期提示一级"] == "退票"
    assert result["预期提示二级"] == "催退票"


def test_parse_markdown_wrapped_json():
    raw = '```json\n{"预测分析":"分析","预期提示一级":"退票","预期提示二级":"催退票","预期提示事件小结":"小结","辅助话术":"话术"}\n```'
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert result["预期提示一级"] == "退票"


def test_parse_json_with_surrounding_text():
    raw = '根据分析，预测如下：\n{"预测分析":"分析","预期提示一级":"退票","预期提示二级":"催退票","预期提示事件小结":"小结","辅助话术":"话术"}\n以上是预测结果。'
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert result["预期提示一级"] == "退票"


def test_parse_strips_brackets_from_level1():
    raw = json.dumps({"预测分析": "分析", "预期提示一级": "【退票】", "预期提示二级": "催退票", "预期提示事件小结": "小结", "辅助话术": "话术"}, ensure_ascii=False)
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert result["预期提示一级"] == "退票"


def test_parse_fuzzy_match_level1():
    raw = json.dumps({"预测分析": "分析", "预期提示一级": "求助投诉", "预期提示二级": "退票费、改签费高", "预期提示事件小结": "小结", "辅助话术": "话术"}, ensure_ascii=False)
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert result["预期提示一级"] == "求助、投诉"


def test_parse_invalid_json_returns_raw():
    raw = "这不是JSON"
    result = parse_prediction(raw, SAMPLE_LABEL_MAPPING)
    assert "raw_output" in result
    assert result["raw_output"] == raw


def test_parse_no_label_mapping():
    raw = json.dumps({"预测分析": "分析", "预期提示一级": "【退票】", "预期提示二级": "催退票", "预期提示事件小结": "小结", "辅助话术": "话术"}, ensure_ascii=False)
    result = parse_prediction(raw, None)
    assert result["预期提示一级"] == "退票"
