# PICO检索API使用指南

## 一、概述

PICO检索已集成到RAGFLOW的检索API中，支持通过API参数启用。本文档说明如何使用PICO检索功能。

## 二、API端点

### 2.1 基础检索API

**端点**: `POST /retrieval`

**位置**: `api/apps/api_app.py`

### 2.2 SDK检索API

**端点**: `POST /retrieval` (SDK)

**位置**: `api/apps/sdk/doc.py`

## 三、请求参数

### 3.1 标准参数（保持不变）

所有标准检索参数仍然有效：

```json
{
  "kb_id": ["knowledgebase_id"],
  "question": "查询问题",
  "page": 1,
  "page_size": 30,
  "similarity_threshold": 0.2,
  "vector_similarity_weight": 0.3,
  "top_k": 1024,
  "highlight": false,
  "doc_ids": [],
  "rerank_id": "rerank_model_id",
  "keyword": false
}
```

### 3.2 PICO检索参数（新增）

#### 策略选择参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `retrieval_strategy` | string | `"standard"` | 检索策略：`"standard"`（标准）、`"pico"`（PICO）、`"auto"`（自动） |
| `use_pico` | boolean | `false` | 是否使用PICO检索（`true`时等同于`retrieval_strategy="pico"`） |

#### PICO配置参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `pico_strict_mode` | boolean | `true` | 严格模式：必须同时满足P+I+O |
| `pico_topk_per_dimension` | integer | `300` | 每个维度（P/I/O）检索的Top-K chunks |
| `pico_max_chunks_per_doc` | integer | `10` | 每篇文章最多返回的chunks数 |

#### UMLS配置参数（可选）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `pico_umls_enabled` | boolean | `false` | 是否启用UMLS同义词映射 |
| `pico_umls_api_key` | string | - | UMLS API key（需要UMLS账号） |
| `pico_umls_base_url` | string | `"https://uts-ws.nlm.nih.gov"` | UMLS API基础URL |

## 四、使用示例

### 4.1 启用PICO检索（显式指定）

```bash
curl -X POST "http://localhost:9380/retrieval" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": ["your_kb_id"],
    "question": "糖尿病患者使用二甲双胍能否降低血糖？",
    "retrieval_strategy": "pico",
    "pico_strict_mode": true,
    "pico_topk_per_dimension": 300,
    "pico_max_chunks_per_doc": 10
  }'
```

### 4.2 使用use_pico参数

```bash
curl -X POST "http://localhost:9380/retrieval" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": ["your_kb_id"],
    "question": "糖尿病患者使用二甲双胍能否降低血糖？",
    "use_pico": true,
    "pico_strict_mode": false
  }'
```

### 4.3 自动策略选择

```bash
curl -X POST "http://localhost:9380/retrieval" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": ["your_kb_id"],
    "question": "糖尿病患者使用二甲双胍能否降低血糖？",
    "retrieval_strategy": "auto"
  }'
```

系统会自动判断是否为循证医学查询，如果是则使用PICO检索，否则使用标准检索。

### 4.4 带UMLS映射的PICO检索

```bash
curl -X POST "http://localhost:9380/retrieval" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": ["your_kb_id"],
    "question": "糖尿病患者使用二甲双胍能否降低血糖？",
    "retrieval_strategy": "pico",
    "pico_umls_enabled": true,
    "pico_umls_api_key": "your_umls_api_key"
  }'
```

### 4.5 Python SDK示例

```python
import requests

url = "http://localhost:9380/retrieval"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "kb_id": ["your_kb_id"],
    "question": "糖尿病患者使用二甲双胍能否降低血糖？",
    "retrieval_strategy": "pico",
    "pico_strict_mode": True,
    "pico_topk_per_dimension": 300
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

# 结果包含标准检索字段
chunks = result["data"]["chunks"]
total = result["data"]["total"]

# 如果使用PICO检索，还会包含pico_info字段
if "pico_info" in result["data"]:
    pico_info = result["data"]["pico_info"]
    print(f"P: {pico_info.get('P')}")
    print(f"I: {pico_info.get('I')}")
    print(f"O: {pico_info.get('O')}")
```

## 五、响应格式

### 5.1 标准响应格式

响应格式与标准检索API完全兼容：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "total": 100,
    "chunks": [
      {
        "chunk_id": "chunk_id_1",
        "doc_id": "doc_id_1",
        "content_with_weight": "chunk content...",
        "similarity": 0.95,
        "docnm_kwd": "document_name"
      }
    ],
    "doc_aggs": {}
  }
}
```

### 5.2 PICO检索额外字段

如果使用PICO检索，响应中会包含`pico_info`字段：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "total": 50,
    "chunks": [...],
    "doc_aggs": {},
    "pico_info": {
      "P": "糖尿病患者",
      "I": "二甲双胍",
      "O": "降低血糖",
      "C": null
    }
  }
}
```

## 六、策略选择逻辑

### 6.1 优先级

1. **显式指定策略**：如果指定了`retrieval_strategy`，直接使用该策略
2. **use_pico参数**：如果`use_pico=true`，使用PICO检索
3. **自动判断**：如果`retrieval_strategy="auto"`，系统自动判断
4. **默认**：默认使用标准检索

### 6.2 自动判断规则

自动判断基于关键词匹配：

- 包含以下关键词2个及以上时，判定为循证医学查询：
  - "能否"、"是否"、"效果"、"疗效"、"治疗"、"干预"
  - "患者"、"人群"、"结局"、"结果"、"对比"、"比较"
  - "预防"、"改善"、"降低"、"提高"、"减少"、"增加"

### 6.3 回退机制

如果PICO检索失败（如LLM提取失败、检索结果为空等），系统会自动回退到标准检索，确保API正常响应。

## 七、性能优化建议

### 7.1 参数调优

- **pico_topk_per_dimension**: 
  - 值越大，召回率越高，但延迟也越高
  - 推荐范围：200-500
  - 默认值：300

- **pico_max_chunks_per_doc**:
  - 控制每篇文章返回的chunks数
  - 推荐范围：5-15
  - 默认值：10

- **pico_strict_mode**:
  - `true`: 严格模式，必须同时满足P+I+O，精度高但可能召回率低
  - `false`: 宽松模式，允许P+I或I+O，召回率高但精度可能降低

### 7.2 缓存

目前PICO提取结果缓存功能未集成到API中，后续版本可能会添加。如果需要，可以通过Redis等缓存系统自行实现。

### 7.3 异步处理

对于批量检索场景，建议：
- 使用异步请求（如果API支持）
- 合理控制并发数
- 设置合适的超时时间

## 八、错误处理

### 8.1 常见错误

1. **PICO模块不可用**
   - 错误：`PICO retrieval module not available`
   - 处理：检查`rag/retrieval/pico_retriever.py`是否存在
   - 回退：自动使用标准检索

2. **PICO提取失败**
   - 错误：LLM提取PICO结构失败
   - 处理：自动回退到标准检索
   - 检查：查看日志中的错误信息

3. **检索结果为空**
   - 可能原因：
    - 严格模式下没有文献同时满足P+I+O
    - 同义词扩展不足
   - 处理：
    - 尝试宽松模式（`pico_strict_mode=false`）
    - 增加`pico_topk_per_dimension`
    - 检查查询是否为循证医学问题

### 8.2 日志查看

PICO检索的相关日志会在以下位置：

- API日志：查看RAGFLOW服务日志
- PICO提取：日志中包含`PICO提取`关键词
- 检索过程：日志中包含`PICO检索`关键词
- 回退信息：日志中包含`fallback to standard retrieval`关键词

## 九、最佳实践

### 9.1 适用场景

✅ **推荐使用PICO检索**：
- 循证医学查询
- 临床决策支持
- 医学研究检索
- 需要证据完整性的场景

⚠️ **谨慎使用**：
- 简单事实性查询
- 非循证医学问题
- 对延迟敏感的场景

### 9.2 查询优化

1. **明确表达P/I/O要素**：
   - 好的查询："糖尿病患者使用二甲双胍能否降低血糖？"
   - 模糊查询："二甲双胍对糖尿病有用吗？"

2. **使用医学标准术语**：
   - 有助于提高同义词扩展质量
   - 例如："2型糖尿病"优于"二型糖尿病"

3. **避免过于复杂的查询**：
   - PICO提取可能无法处理过于复杂或包含多个研究问题的查询

### 9.3 参数配置建议

**高精度场景**（严格模式）：
```json
{
  "pico_strict_mode": true,
  "pico_topk_per_dimension": 300,
  "pico_max_chunks_per_doc": 10
}
```

**高召回场景**（宽松模式）：
```json
{
  "pico_strict_mode": false,
  "pico_topk_per_dimension": 500,
  "pico_max_chunks_per_doc": 15
}
```

**快速响应场景**：
```json
{
  "pico_strict_mode": true,
  "pico_topk_per_dimension": 200,
  "pico_max_chunks_per_doc": 5
}
```

## 十、版本兼容性

- **RAGFLOW版本**: 要求支持动态检索策略选择
- **Python版本**: Python 3.8+
- **依赖**: 
  - 如果PICO模块不可用，API仍可正常工作（自动回退）

## 十一、更新日志

### v1.0.0 (当前版本)
- ✅ 集成PICO检索到基础检索API
- ✅ 集成PICO检索到SDK检索API
- ✅ 支持策略选择（standard/pico/auto）
- ✅ 支持PICO配置参数
- ✅ 支持UMLS映射（可选）
- ✅ 自动回退机制

## 十二、技术支持

如有问题或建议，请：
1. 查看日志定位问题
2. 参考[PICO检索实现指南](./PICO_Retrieval_Implementation_Guide.md)
3. 参考[PICO检索评估文档](./PICO_RAG_Evaluation.md)



