"""请求/响应 Pydantic 模型 - 字段统一使用驼峰命名（SOA契约规范）"""
from typing import Optional
from pydantic import BaseModel


class IntentPredictionRequest(BaseModel):
    orderId: int
    sessionId: str
    language: Optional[str] = None  # ISO 639-1 语言代码(en/ja/ko/th等)，为空则默认中文


class PredictionContext(BaseModel):
    orderId: int
    relatedOrderIds: list[int]
    sessionId: str
    summaryCount: int
    blockedRecordCount: int
    ubtRecordCount: int = 0
    notificationCount: int = 0
    complaintCount: int = 0
    implusMessageCount: int = 0
    ttsRecordCount: int = 0
    hasOrderStatus: bool = False
    queryTimeRange: dict


class PredictionResult(BaseModel):
    predictionAnalysis: str
    level1Label: str
    level2Label: str
    eventSummary: str
    suggestedScript: str
    context: PredictionContext


class IntentPredictionResponse(BaseModel):
    success: bool
    data: Optional[PredictionResult] = None
    errorMessage: Optional[str] = None
    processingTime: float
