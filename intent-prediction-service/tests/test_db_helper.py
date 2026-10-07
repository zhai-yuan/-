"""测试预测结果落库 SQL 生成"""
from app.db_helper import DbHelper
from app.models import PredictionContext, PredictionResult


class FakeMysqlPool:
    def __init__(self):
        self.sql = ""
        self._select_results: list[list[dict]] = []

    def execute(self, sql: str) -> None:
        self.sql = sql

    def select(self, sql: str) -> list[dict]:
        self.sql = sql
        if self._select_results:
            return self._select_results.pop(0)
        return []


def _prediction_result() -> PredictionResult:
    context = PredictionContext(
        orderId=123,
        relatedOrderIds=[123, 456],
        sessionId="s1",
        summaryCount=2,
        blockedRecordCount=0,
        ubtRecordCount=1,
        queryTimeRange={"start": "2026-04-08T00:00:00", "end": "2026-04-15T23:59:59"},
    )
    return PredictionResult(
        predictionAnalysis="分析内容",
        level1Label="退票",
        level2Label="催退款",
        eventSummary="客户催促退款进度",
        suggestedScript="您好，请问是咨询退款进度吗？",
        context=context,
    )


def test_insert_prediction_result_uses_insert_ignore():
    """测试落表 SQL 使用 INSERT IGNORE（不覆盖已有记录）"""
    fake_pool = FakeMysqlPool()
    helper = object.__new__(DbHelper)
    helper.mysql_pool = fake_pool

    ok = helper.insert_prediction_result(
        _prediction_result(),
        "## 标签体系\n退票\n用户说：'退款什么时候到账？'",
    )

    assert ok is True
    assert "INSERT IGNORE INTO" in fake_pool.sql
    assert "ON DUPLICATE KEY UPDATE" not in fake_pool.sql
    assert "`userPrompt`" in fake_pool.sql
    assert "用户说：\\'退款什么时候到账？\\'" in fake_pool.sql


def test_query_prediction_by_session_returns_row():
    """测试去重查询：命中时返回 dict"""
    fake_pool = FakeMysqlPool()
    helper = object.__new__(DbHelper)
    helper.mysql_pool = fake_pool
    fake_pool._select_results = [
        [{"orderId": 123, "sessionId": "s1", "level1Label": "退票", "level2Label": "催退款",
          "predictionAnalysis": "分析", "eventSummary": "小结", "suggestedScript": "话术",
          "relatedOrderIds": "[123, 456]"}]
    ]

    row = helper.query_prediction_by_session("s1")

    assert row is not None
    assert row["orderId"] == 123
    assert row["level1Label"] == "退票"
    assert "sessionId = 's1'" in fake_pool.sql


def test_query_prediction_by_session_returns_none_when_empty():
    """测试去重查询：无记录时返回 None"""
    fake_pool = FakeMysqlPool()
    helper = object.__new__(DbHelper)
    helper.mysql_pool = fake_pool
    fake_pool._select_results = [[]]

    row = helper.query_prediction_by_session("nonexistent")

    assert row is None


def test_query_prediction_by_session_returns_none_on_exception():
    """测试去重查询：异常时返回 None（降级）"""
    class ErrorPool:
        def select(self, sql: str):
            raise RuntimeError("DB connection lost")

    helper = object.__new__(DbHelper)
    helper.mysql_pool = ErrorPool()

    row = helper.query_prediction_by_session("s1")

    assert row is None
