# 进线意图预测服务 (Intent Prediction Service)

基于客户历史会话摘要、订单状态、系统通知、投诉单、在线会话记录及用户行为数据，预测客户下次进线的意图，输出预期提示一级、二级、事件小结、预测分析、辅助话术五个维度的结构化信息。

## 数据获取流程图

```
输入: orderId + sessionId
│
├─────────────────── 层1: 并行获取（7路并发）──────────────────────────┐
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ [主订单] get_related_order_ids (api/11817)                   │    │
│  │  → 返回: 关联订单ID集合 + 原始响应                            │    │
│  │  → 副产品: extract_order_status() 零成本提取订单状态          │    │
│  │    (退票/改签/航变 来自同一个API响应，无额外调用)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ [主订单] get_notification_timeline                            │    │
│  │  内部再并行3路:                                               │    │
│  │    ├── 站内信 (api/15098/queryLetterHistoryV2)               │    │
│  │    ├── 微信   (api/14464/queryMessage)                       │    │
│  │    └── APP推送 (api/15098/queryAppHistory)                   │    │
│  │  → 标准化 + 跨渠道去重 → 统一通知时间线 (最多10条)           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────┐                               │
│  │ [主订单] search_complaints        │                               │
│  │  (api/14718/searchComplaint)      │                               │
│  │  → 返回所有投诉(含已结案)         │                               │
│  └──────────────────────────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────┐                               │
│  │ [主订单] get_implus_messages      │                               │
│  │  (api/15408/getMessagesByOrderId) │                               │
│  │  → 仅文本消息, 最近1个session     │                               │
│  │  → 最多5条                        │                               │
│  └──────────────────────────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────┐                               │
│  │ [主订单] query_user_blocked       │                               │
│  │  (api/11504)                      │                               │
│  └──────────────────────────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────┐                               │
│  │ [主订单] query_order_detail_ubt   │                               │
│  │  (MyTrix SQL引擎)                 │                               │
│  └──────────────────────────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────┐                               │
│  │ [主订单] get_tts_outcall_results  │                               │
│  │  (api/14242/QueryFlightOutCall    │                               │
│  │   Result)                         │                               │
│  │  → IVR外呼TTS通知(航变/取消等)   │                               │
│  │  → 最多5条                        │                               │
│  └──────────────────────────────────┘                               │
│                                                                      │
├─────────────────── 层2: 并行查进线/外呼记录 ─────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ [主订单 + 关联子订单] 对每个 order_id:                        │    │
│  │    ├── get_incomecall_records (api/14537)                    │    │
│  │    └── get_outcall_records   (api/14537)                    │    │
│  │  → 收集 callId + sessionId                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
├─────────────────── 层3: 并行查摘要 ──────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 对每个 callId:                                               │    │
│  │    └── query_tel_summary  (api/17307)                       │    │
│  │ 对每个 chat sessionId:                                       │    │
│  │    └── query_chat_summary (api/15408)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
│
▼
Prompt 组装 (按信号强度排序)
┌──────────────────────────────────────────┐
│ ## 标签体系                               │
│ ## 订单当前状态         ← 退票/改签/航变   │
│ ## 历史交互摘要（时间正序）               │
│ ## 系统通知记录（时间倒序）← 去重后max10  │
│ ## IVR外呼通知记录（时间倒序）← TTS max5 │
│ ## 活跃投诉单                             │
│ ## 在线会话记录         ← IMPlus最近对话  │
│ ## 用户受阻记录                           │
│ ## 订单详情页浏览记录                     │
└──────────────────────────────────────────┘
│
▼
LLM 推理 → 解析 → 返回结构化预测结果
```

### 同 Session 去重机制

同一次进线会话（sessionId）可能因客服多次点击订单详情页触发多次预测请求。为保证首次预测结果不被后续数据变化污染，服务实现了请求级去重：

```
请求进来 → 获取 sessionId 进程内锁
  │
  ├── 查 DB (idx_sessionId 索引命中)
  │     ├── 有记录 → 直接返回历史预测结果 (~10ms)
  │     └── 无记录 → 执行完整预测流程 → INSERT IGNORE 落表
  │
  └── 释放锁
```

**设计要点：**
- 进程内 `asyncio.Lock`（per sessionId）防止同实例并发穿透
- 多实例（多 Pod）竞态由 MySQL `INSERT IGNORE` + UNIQUE KEY 兜底，只保留首条
- QConfig 开关 `features.dedup_enabled`，可动态关闭回退到原有行为
- DB 查询失败时自动降级为正常预测流程（不影响可用性）

### 接口调用范围说明

| 数据源 | 调用范围 | 说明 |
|--------|----------|------|
| 订单状态(退票/改签/航变) | 主订单 | 复用 get_related_order_ids 的 API 响应，零额外调用 |
| 站内信/微信/APP推送 | 主订单 | 三渠道去重合并 |
| IVR外呼TTS通知 | 主订单 | 航变/取消/改签等语音通知内容及接听状态 |
| 投诉单 | 主订单 | 含活跃和已结案，帮助LLM判断投诉是否仍在进行 |
| IMPlus在线会话 | 主订单 | 仅最近session的文本消息 |
| 用户受阻记录 | 主订单 | — |
| UBT浏览埋点 | 主订单 | — |
| 进线/外呼记录 | 主订单 + 关联子订单 | 唯一查关联订单的接口 |
| 电话/会话摘要 | 按callId/sessionId | 依赖层2的结果 |

> **设计原则**: 新增数据源（通知/投诉/IMPlus/订单状态）仅对主订单调用，不扩散到关联子订单。
> 只有进线/外呼记录需要查关联订单（因为用户可能用关联订单进线过）。

## 项目结构

```
intent-prediction-service/
├── main.py                  # 生产入口 (uvicorn, port 8080)
├── start_service.py         # 开发入口 (reload, port 8000)
├── requirements.txt
├── .paas/                   # Captain 部署配置
├── app/
│   ├── main.py              # FastAPI 路由 + SOA 兼容
│   ├── models.py            # Pydantic 请求/响应模型
│   ├── db_helper.py         # 异步落表
│   └── service.py           # 核心编排逻辑
├── clients/
│   ├── summary_client.py    # SOA API 封装 (11817/14537/17307/15408/11504)
│   ├── llm_client.py        # LangChain LLM 调用 + fallback + 输出解析
│   ├── ubt_client.py        # UBT 埋点查询 (MyTrix SQL引擎)
│   ├── notification_client.py  # 站内信+微信+APP推送 (去重合并)
│   ├── complaint_client.py     # 投诉单查询
│   ├── implus_client.py        # IMPlus在线会话
│   ├── tts_client.py           # IVR外呼TTS通知查询
│   └── order_status_extractor.py # 订单状态提取 (退票/改签/航变)
├── config/
│   ├── settings.py          # QConfig 动态配置 (线上修改立即生效)
│   └── prompts.py           # Prompt 模板 + 格式化函数
├── data/
│   └── label_mapping.json   # 标签体系 (8个一级, 80+个二级)
├── TEMP/
│   └── qconfig/             # QConfig 配置文件 (上传到线上)
│       ├── base_config.json
│       └── prompt_config.json
├── example/
│   └── call_service.py      # 本地调用示例脚本
└── tests/
```

## 本地开发

### 1. 安装依赖

```bash
cd intent-prediction-service
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python start_service.py
```

服务启动后：
- 服务地址: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 开发模式自带热重载，修改代码后自动重启

### 3. 调用示例

使用 `example/call_service.py` 快速测试：

```bash
python example/call_service.py --order-id 1128146287853336 --session-id 06562136267814598188
```

或使用 curl：

```bash
curl -X POST http://localhost:8000/intentPrediction \
  -H "Content-Type: application/json" \
  -d '{"orderId": 1658113007166537, "sessionId": "06562136267814598188", "language": "en"}'
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## API

### 意图预测

```
POST /intentPrediction
POST /bjjson/intentPrediction   (SOA 兼容)
```

**请求：**

```json
{
  "orderId": 1128147844465509,
  "sessionId": "06562136267814598188"
}
```

**响应：**

```json
{
  "success": true,
  "data": {
    "predictionAnalysis": "客人多次因病退材料日期不一致被驳回...",
    "level1Label": "退票",
    "level2Label": "通知材料上传",
    "eventSummary": "客户来电咨询病退材料审核结果及后续所需材料",
    "suggestedScript": "您好，注意到您郑州-丽江行程有提交病退申请，请问是咨询退票材料的相关问题吗？",
    "context": {
      "orderId": 1128146287853336,
      "relatedOrderIds": [1128146287823192, 1128146287853336],
      "sessionId": "06562136267814598188",
      "summaryCount": 12,
      "blockedRecordCount": 0,
      "queryTimeRange": {"start": "2026-04-08T00:00:00", "end": "2026-04-15T23:59:59"}
    }
  },
  "errorMessage": null,
  "processingTime": 3.217
}
```

### 批量评估（T+1 准确率回检）

```
POST /admin/triggerEvaluation
```

对前一天的预测结果进行 T+1 准确率评估：从 Hive 获取真实通话内容，用 LLM 判断预测是否准确，结果回写 MySQL。

#### 运行模式

| 模式 | 触发方式 | 行为 |
|------|----------|------|
| Zeus 定时调度（全量） | 不传 `limit` | 前置检查 → 通过后提交后台异步执行，接口立即返回 |
| 手动调试（少量） | 传 `limit` | 同步执行，等待完成后返回完整结果 |

#### 请求参数

参数可通过 **Query Params** 或 **JSON Body** 传递（优先读 query params）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `date` | string (YYYY-MM-DD) | 否 | 评估目标日期，默认为昨天 |
| `limit` | int | 否 | 最大处理条数；传此参数进入同步调试模式 |

#### 前置条件

- QConfig 配置项 `evaluation_manual_trigger_enabled` 必须为 `true`，否则返回 HTTP 403
- Hive 分区 `dwflt.cdm_service_cdr_contact_msg_di` 对应日期必须有数据
- MySQL `intent_prediction_results` 表中该日期存在 `isAccurate IS NULL` 的待评估记录

#### 调用示例

**Zeus 全量调度（异步模式）：**

```bash
curl -X POST http://intent-prediction.flight.ctripgroup.cn/admin/triggerEvaluation
```

**指定日期全量：**

```bash
curl -X POST "http://intent-prediction.flight.ctripgroup.cn/admin/triggerEvaluation?date=2026-05-30"
```

**手动调试（同步模式，限制 10 条）：**

```bash
curl -X POST http://intent-prediction.flight.ctripgroup.cn/admin/triggerEvaluation \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-30", "limit": 100}'
```

#### 响应示例

**异步模式 - 提交成功：**

```json
{"success": true, "message": "批量评估任务已提交后台执行"}
```

**异步模式 - 前置检查失败（Zeus 会重试）：**

```json
{"success": false, "message": "Hive 分区 dt=2026-05-30 无数据，可能尚未就绪"}
```

**同步模式 - 返回完整统计：**

```json
{
  "success": true,
  "stats": {
    "total": 10,
    "evaluated": 8,
    "accurate": 6,
    "inaccurate": 2,
    "skipped": 2,
    "accuracy_rate": 0.75
  }
}
```

#### 执行流程

```
触发请求
│
├─ 有 limit? ─── 是 ──→ 同步执行 run_batch_evaluation() → 返回统计
│
└─ 无 limit（Zeus调度）
    │
    ├─ Step 1: preflight_check()
    │   ├─ 查 MySQL 是否有待评估记录
    │   └─ 查 Hive 分区是否就绪
    │       ├─ 失败 → 返回 {success: false} (Zeus 重试)
    │       └─ 通过 ↓
    │
    └─ Step 2: asyncio.ensure_future(run_batch_evaluation())
        │
        ├─ 查 MySQL 获取待评估预测记录 (isAccurate IS NULL)
        ├─ 按 sessionId 批量查 Hive 获取真实通话内容
        ├─ 逐条调用 LLM 判断准确性 + 可预测性分析 (并发度: evaluation_concurrency)
        │   ├─ 主模型: vLLM Qwen3.5-27B-FP8
        │   └─ 降级模型: aigw Qwen3.5-27B
        └─ 结果回写 MySQL (isAccurate, callSummary, isPredictable, supportingSignals, missingInfo)
```

#### 可预测性分析

评估流程在判断预测准确性的同时，还会分析该真实意图**是否有可能**基于预测时的可用信息被预测出来。这用于区分"模型能力不足"和"信息本身不足"两种失败原因，帮助计算更有意义的"有效准确率"。

| 字段 | 值域 | 说明 |
|------|------|------|
| `isPredictable` | 1/2/0/NULL | 1=可预测, 2=部分可预测, 0=不可预测, NULL=未分析 |
| `supportingSignals` | text | 支持预测的信号（多条用｜分隔） |
| `missingInfo` | text | 缺失的关键信息（多条用｜分隔） |

**统计查询示例 - 有效准确率（排除不可预测case）：**

```sql
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN isAccurate = 1 THEN 1 ELSE 0 END) as accurate,
  SUM(CASE WHEN isPredictable = 0 THEN 1 ELSE 0 END) as unpredictable,
  ROUND(SUM(CASE WHEN isAccurate = 1 THEN 1 ELSE 0 END) * 100.0
    / (COUNT(*) - SUM(CASE WHEN isPredictable = 0 THEN 1 ELSE 0 END)), 2) as effective_accuracy_pct
FROM intent_prediction_results
WHERE isAccurate IS NOT NULL AND DATE(DataChange_CreateTime) = '2026-06-11';
```

#### 相关配置（QConfig）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `evaluation_manual_trigger_enabled` | `false` | 是否允许触发评估 |
| `evaluation_concurrency` | `5` | LLM 并发调用数 |
| `evaluation_batch_size` | `500` | Hive 查询批次大小 |
| `evaluation_llm_model` | `Qwen3.5-27B-FP8` | 主评估模型 |
| `evaluation_hive_table` | `dwflt.cdm_service_cdr_contact_msg_di` | Hive 数据源表 |

### 辅助接口

| 接口 | 用途 |
|------|------|
| `GET /` | 服务信息 |
| `GET /health` | 健康检查 |
| `GET /checkhealth.json` | SOA 健康探针 |
| `GET /_operationinfo` | SOA 操作描述 |
| `GET /stats` | 请求统计 |

## 部署

通过 Captain PaaS 发布，`.paas/` 目录包含完整的部署配置：
- `uvicorn.ini`: supervisord 配置，端口 8080
- `on_image.sh`: 镜像构建脚本（安装 uvicorn + bicommonkits）

配置项通过 QConfig 远程管理（appid: 100072638），所有配置修改后立即生效，无需重启服务。
