"""
核心编排逻辑
并行获取数据 -> 拼装 prompt -> LLM 推理 -> 解析结果
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

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

from app.models import (
    IntentPredictionResponse,
    PredictionResult,
    PredictionContext,
)
from app.db_helper import get_db_helper
from clients import summary_client, ubt_client
from clients import notification_client, complaint_client, implus_client, tts_client
from clients.order_status_extractor import extract_order_status
from clients.llm_client import call_llm, parse_prediction
from config.settings import settings
from config.prompts import (
    get_system_prompt,
    build_user_prompt,
    build_label_reference,
    build_language_instruction,
    format_summaries,
    format_blocked_records,
    format_ubt_records,
    format_order_status,
    format_notification_timeline,
    format_complaints,
    format_implus_messages,
    format_tts_records,
)


# 启动时加载标签体系到内存
_label_mapping: dict | None = None


def _get_label_mapping() -> dict:
    global _label_mapping
    if _label_mapping is None:
        path = Path(__file__).parent.parent / "data" / "label_mapping.json"
        with open(path, "r", encoding="utf-8") as f:
            _label_mapping = json.load(f)
    return _label_mapping


class IntentPredictionService:
    """意图预测服务"""

    def __init__(self):
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._cache_hit_count = 0
        # 进程内锁：同一 sessionId 的并发请求串行化，防止同实例内重复 LLM 调用
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """获取或创建 session 级别的锁"""
        async with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    async def _remove_session_lock(self, session_id: str) -> None:
        """预测完成后清理锁，避免内存泄漏"""
        async with self._locks_lock:
            self._session_locks.pop(session_id, None)

    async def predict(self, order_id: int, session_id: str, language: str = None) -> IntentPredictionResponse:
        """
        执行完整预测流程（含去重）:
        0. 查 DB 是否已有该 session 的预测结果（去重）
        1. 并行获取历史数据
        2. 拼装 prompt
        3. LLM 推理
        4. 解析结果
        """
        start_time = time.time()
        self._request_count += 1
        tags = {"orderId": str(order_id), "sessionId": session_id}

        # ---- 去重逻辑 ----
        if settings.dedup_enabled:
            lock = await self._get_session_lock(session_id)
            async with lock:
                # 查 DB：是否已有该 session 的预测结果
                cached = await self._try_get_cached(session_id, tags)
                if cached is not None:
                    self._cache_hit_count += 1
                    elapsed = time.time() - start_time
                    cached.processingTime = round(elapsed, 3)
                    LogHelper.log_info(
                        f"去重命中，返回历史结果: sessionId={session_id}", tags
                    )
                    return cached
                # 未命中 → 执行完整预测（仍在锁内，防止同实例并发穿透）
                result = await self._do_predict(order_id, session_id, start_time, tags, language=language)
            await self._remove_session_lock(session_id)
            return result
        else:
            return await self._do_predict(order_id, session_id, start_time, tags, language=language)

    async def _try_get_cached(
        self, session_id: str, tags: dict
    ) -> IntentPredictionResponse | None:
        """从 DB 查询历史预测结果，构造响应。查询失败时返回 None（降级为正常预测）。"""
        try:
            row = await asyncio.to_thread(
                get_db_helper().query_prediction_by_session, session_id
            )
        except Exception:
            return None

        if row is None:
            return None

        related_ids = []
        raw_related = row.get("relatedOrderIds")
        if raw_related:
            try:
                related_ids = json.loads(raw_related)
            except Exception:
                pass

        context = PredictionContext(
            orderId=int(row.get("orderId", 0)),
            relatedOrderIds=related_ids,
            sessionId=session_id,
            summaryCount=0,
            blockedRecordCount=0,
            queryTimeRange={},
        )

        result = PredictionResult(
            predictionAnalysis=row.get("predictionAnalysis", ""),
            level1Label=row.get("level1Label", ""),
            level2Label=row.get("level2Label", ""),
            eventSummary=row.get("eventSummary", ""),
            suggestedScript=row.get("suggestedScript", ""),
            context=context,
        )

        return IntentPredictionResponse(
            success=True,
            data=result,
            processingTime=0,
        )

    async def _do_predict(
        self, order_id: int, session_id: str, start_time: float, tags: dict, language: str = None
    ) -> IntentPredictionResponse:
        """执行完整预测流程（数据获取 → prompt → LLM → 解析 → 落表）"""
        try:
            # ---- Step 1: 并行数据获取 ----
            t_fetch = time.perf_counter()
            fetch_result = await self._fetch_data(order_id, session_id)
            (summaries, blocked_records, ubt_records, all_order_ids,
             query_range, notifications, order_statuses, complaints, implus_messages, tts_records) = fetch_result
            fetch_elapsed = time.perf_counter() - t_fetch
            LogHelper.log_info(
                f"数据获取完成: {len(summaries)} 条摘要, {len(blocked_records)} 条受阻记录, "
                f"{len(ubt_records)} 条浏览埋点, {len(notifications)} 条通知, "
                f"{len(complaints)} 条投诉, {len(implus_messages)} 条会话消息, "
                f"{len(tts_records)} 条TTS外呼, "
                f"耗时 {fetch_elapsed:.3f}s",
                tags,
            )

            # 触发预测的维度：摘要、通知、投诉、IMPlus 会话、TTS 外呼
            # 排除「受阻记录」和「浏览埋点」——它们只是辅助信号，不足以单独驱动 LLM
            has_signal = bool(
                summaries
                or notifications
                or complaints
                or implus_messages
                or tts_records
            )
            if not has_signal:
                self._error_count += 1
                elapsed = time.time() - start_time
                return IntentPredictionResponse(
                    success=False,
                    errorMessage="无足够历史数据进行预测",
                    processingTime=round(elapsed, 3),
                )

            # ---- Step 2: 拼装 prompt ----
            label_mapping = _get_label_mapping()
            label_ref = build_label_reference(label_mapping)
            summaries_text = format_summaries(summaries)
            blocked_text = format_blocked_records(blocked_records)
            ubt_text = format_ubt_records(ubt_records)
            order_status_text = format_order_status(order_statuses)
            notification_text = format_notification_timeline(notifications)
            complaint_text = format_complaints(complaints)
            implus_text = format_implus_messages(implus_messages)
            tts_text = format_tts_records(tts_records)

            system_prompt = get_system_prompt() + build_language_instruction(language)
            user_prompt = build_user_prompt(
                label_ref, summaries_text, blocked_text, ubt_text,
                formatted_order_status=order_status_text,
                formatted_notifications=notification_text,
                formatted_complaints=complaint_text,
                formatted_implus=implus_text,
                formatted_tts=tts_text,
            )

            # ---- Step 3: LLM 推理 ----
            t_llm = time.perf_counter()
            raw_output = await asyncio.to_thread(call_llm, system_prompt, user_prompt)
            llm_elapsed = time.perf_counter() - t_llm
            LogHelper.log_info(f"LLM 推理完成, 耗时 {llm_elapsed:.3f}s", tags)

            # ---- Step 4: 解析结果 ----
            parsed = parse_prediction(raw_output, label_mapping, language=language)

            if "raw_output" in parsed:
                self._error_count += 1
                elapsed = time.time() - start_time
                return IntentPredictionResponse(
                    success=False,
                    errorMessage=f"LLM 输出解析失败: {parsed['raw_output'][:200]}",
                    processingTime=round(elapsed, 3),
                )

            # 提取置信度
            confidence = parsed.get("置信度", 50)
            is_unpredictable = parsed.get("预期提示一级") == "无法预测"

            context = PredictionContext(
                orderId=order_id,
                relatedOrderIds=sorted(all_order_ids),
                sessionId=session_id,
                summaryCount=len(summaries),
                blockedRecordCount=len(blocked_records),
                ubtRecordCount=len(ubt_records),
                notificationCount=len(notifications),
                complaintCount=len(complaints),
                implusMessageCount=len(implus_messages),
                ttsRecordCount=len(tts_records),
                hasOrderStatus=bool(order_statuses),
                queryTimeRange=query_range,
            )

            result = PredictionResult(
                predictionAnalysis=parsed.get("预测分析", ""),
                level1Label=parsed.get("预期提示一级", ""),
                level2Label=parsed.get("预期提示二级", ""),
                eventSummary=parsed.get("预期提示事件小结", ""),
                suggestedScript=parsed.get("辅助话术", ""),
                context=context,
            )

            self._success_count += 1
            elapsed = time.time() - start_time

            # 异步落表（始终存储，含 confidence）, 不阻塞响应
            asyncio.create_task(self._persist_result(result, user_prompt, confidence))

            # 置信度过滤：低置信度或无法预测时，不下发给客服
            if settings.confidence_enabled and (is_unpredictable or confidence < settings.confidence_threshold):
                LogHelper.log_info(
                    f"置信度过滤: confidence={confidence}, unpredictable={is_unpredictable}, "
                    f"threshold={settings.confidence_threshold}",
                    tags,
                )
                return IntentPredictionResponse(
                    success=False,
                    errorMessage="置信度不足，不输出预测",
                    processingTime=round(elapsed, 3),
                )

            return IntentPredictionResponse(
                success=True,
                data=result,
                processingTime=round(elapsed, 3),
            )

        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            LogHelper.log_error(f"预测异常: {e}", tags)
            return IntentPredictionResponse(
                success=False,
                errorMessage=f"预测失败: {str(e)}",
                processingTime=round(elapsed, 3),
            )

    async def _fetch_data(
        self, order_id: int, session_id: str
    ) -> tuple[list, list, list, set, dict, list, list, list, list, list]:
        """
        并行数据获取:
        层1: 反查关联订单(+订单状态) + 用户受阻 + UBT + 通知时间线 + 投诉单 + IMPlus + TTS外呼
        层2: 对每个订单查进线/外呼记录
        层3: 对每个 callId/sessionId 查摘要
        """
        now = datetime.now()
        start_dt = now - timedelta(days=settings.query_days)
        start_time = start_dt.strftime("%Y-%m-%dT00:00:00")
        end_time = now.strftime("%Y-%m-%dT23:59:59")
        query_range = {"start": start_time, "end": end_time}

        # 通知时间用更友好的格式
        notify_start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        notify_end = now.strftime("%Y-%m-%d %H:%M:%S")

        # ---- 层1: 并行 ----
        ubt_task = (
            ubt_client.query_order_detail_ubt(order_id)
            if settings.ubt_enabled
            else asyncio.sleep(0, result=[])
        )
        blocked_task = (
            summary_client.query_user_blocked_records(order_id)
            if settings.blocked_records_enabled
            else asyncio.sleep(0, result=[])
        )
        notification_task = (
            notification_client.get_notification_timeline(order_id, notify_start, notify_end)
            if settings.notifications_enabled
            else asyncio.sleep(0, result=[])
        )
        complaint_task = (
            complaint_client.search_complaints(order_id)
            if settings.complaint_enabled
            else asyncio.sleep(0, result=[])
        )
        implus_task = (
            implus_client.get_implus_messages(order_id)
            if settings.implus_enabled
            else asyncio.sleep(0, result=[])
        )
        tts_task = (
            tts_client.get_tts_outcall_results(order_id)
            if settings.tts_enabled
            else asyncio.sleep(0, result=[])
        )

        (all_order_ids, raw_order_resp), blocked_records, ubt_records, notifications, complaints, implus_messages, tts_records = (
            await asyncio.gather(
                summary_client.get_related_order_ids(order_id),
                blocked_task,
                ubt_task,
                notification_task,
                complaint_task,
                implus_task,
                tts_task,
            )
        )

        # 零成本提取订单状态（复用已有 API 响应）
        order_statuses = []
        if settings.order_status_enabled and raw_order_resp:
            try:
                order_statuses = extract_order_status(raw_order_resp)
            except Exception as e:
                LogHelper.log_warn(f"订单状态提取失败: {e}", {"orderId": str(order_id), "sessionId": session_id})

        # ---- 层2: 并行查进线/外呼 ----
        record_tasks = []
        for oid in all_order_ids:
            record_tasks.append(summary_client.get_incomecall_records(oid, start_time, end_time))
            record_tasks.append(summary_client.get_outcall_records(oid, start_time, end_time))
        record_results = await asyncio.gather(*record_tasks, return_exceptions=True)

        # 收集 callId 和 sessionId，排除当前 session_id
        call_ids = set()
        chat_session_ids = set()
        for result in record_results:
            if isinstance(result, Exception):
                continue
            for record in result:
                call_id = record.get("callID", "")
                sid = record.get("sessionid", "")
                source = record.get("source", "")
                if source == "在线客服" and sid:
                    chat_session_ids.add(sid)
                elif call_id:
                    call_ids.add(call_id)

        # 过滤当前 session_id - 当前进线的摘要不作为历史
        call_ids.discard(session_id)
        chat_session_ids.discard(session_id)

        # ---- 层3: 并行查摘要 ----
        summary_tasks = []
        for cid in call_ids:
            summary_tasks.append(summary_client.query_tel_summary(cid))
        for sid in chat_session_ids:
            summary_tasks.append(summary_client.query_chat_summary(sid))
        summary_results = await asyncio.gather(*summary_tasks, return_exceptions=True)

        # 汇总摘要
        all_summaries = []
        for result in summary_results:
            if isinstance(result, Exception):
                continue
            for s in result:
                summary_text = s.get("summary", "")
                if summary_text and summary_text != "无有效总结":
                    all_summaries.append(s)

        # 按 summaryTime 排序
        all_summaries.sort(key=lambda x: x.get("summaryTime", ""))

        return (all_summaries, blocked_records, ubt_records, all_order_ids,
                query_range, notifications, order_statuses, complaints, implus_messages, tts_records)

    def get_stats(self) -> dict:
        return {
            "total_requests": self._request_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "cache_hit_count": self._cache_hit_count,
        }

    async def _persist_result(self, result: PredictionResult, user_prompt: str, confidence: int = 0) -> None:
        """异步落表, 失败仅记录日志, 不影响主流程"""
        try:
            await asyncio.to_thread(
                get_db_helper().insert_prediction_result, result, user_prompt, confidence
            )
        except Exception as e:
            tags = {"orderId": str(result.context.orderId), "sessionId": result.context.sessionId}
            LogHelper.log_error(f"落表任务异常: {e}", tags)

