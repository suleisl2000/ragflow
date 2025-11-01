# PICO框架医疗检索系统实现指南

## 一、概述

本文档说明如何在RAGFLOW中集成和使用PICO框架医疗检索系统。

## 二、核心优势（修正后的评估）

### 2.1 召回率优势 ✅

**相比标准RAG的优势**：
- **多维度独立OR扩展**：每个P/I/O维度可独立扩展同义词，避免跨维度干扰
- **同义词覆盖更广**：结合LLM提取 + UMLS映射 + 现有词典，三层扩展
- **避免术语遗漏**：如"心肌梗死"/"MI"/"heart attack"都能召回

**示例对比**：
```
标准RAG: "糖尿病 二甲双胍 血糖" (单一query，同义词可能互相稀释)

PICO RAG:
  P: "糖尿病 OR 2型糖尿病 OR T2DM OR type 2 diabetes OR 非胰岛素依赖型糖尿病"
  I: "二甲双胍 OR metformin OR 双胍类 OR biguanides"  
  O: "HbA1c OR 糖化血红蛋白 OR 血糖控制 OR glycemic control"
```

### 2.2 术语理解优势 ✅

**LLM vs 分词系统**：
- LLM能理解长尾复合术语："阵发性心房颤动伴快速心室率"
- LLM能处理医学缩写："T2DM"、"MI"
- LLM能识别上下文："患者患有糖尿病，使用二甲双胍治疗"

### 2.3 证据完整性 ✅

- **文章级筛查**：必须同时满足P+I+O的文献才保留
- **避免碎片化**：不会出现"研究A的人群 + 研究B的干预"的错误组合

## 三、架构设计

```
用户查询
  ↓
PICO结构化抽取 (医学LLM + Few-shot)
  ├─ P (Population) - 带同义词扩展
  ├─ I (Intervention) - 带同义词扩展
  ├─ C (Comparison) - 可选，仅作为boost
  └─ O (Outcome) - 带同义词扩展
  ↓
多层容错验证
  ├─ 结构化验证
  ├─ 术语校验
  └─ 回退机制
  ↓
并行三通道检索 (P/I/O独立检索)
  ├─ 每路检索 = 关键词(OR扩展) + 向量 + 融合
  ├─ 同义词来源：LLM提取 + UMLS映射 + 现有词典
  └─ 每路返回Top-K chunks
  ↓
候选chunk聚合与去重
  ↓
文章级PIO一致性筛查
  ├─ 必须同时命中 P + I + O
  ├─ C存在时作为boost加分
  └─ 计算文章级综合评分
  ↓
文章内chunk重排序
  ├─ PIO覆盖密度
  ├─ 邻近性（相邻chunk协同）
  ├─ 章节权重（Results > Methods > Abstract）
  └─ MMR去冗余
  ↓
输出结果（带置信度+证据等级）
```

## 四、代码模块说明

### 4.1 核心类

1. **PICOQueryRewriter**: PICO结构化抽取
   - 使用LLM提取P/I/C/O
   - 带多层容错和回退机制

2. **PICOSynonymExpander**: 同义词扩展
   - LLM提取的同义词
   - UMLS映射（可选）
   - 现有词典（synonym.json + WordNet）

3. **PICOParallelRetriever**: 并行检索
   - 三路并行检索（P/I/O）
   - 每路：关键词(OR) + 向量 + 融合

4. **PIOArticleFilter**: 文章级筛查
   - 必须同时满足P+I+O
   - 支持严格/宽松模式

5. **IntraDocChunkRanker**: 文章内重排序
   - PIO覆盖密度
   - 章节权重
   - 位置权重

6. **PICORetriever**: 主检索类
   - 整合以上所有模块
   - 提供统一的检索接口

### 4.2 关键实现细节

#### 4.2.1 同义词扩展

```python
# 三层扩展策略
1. LLM提取（PICO抽取时已包含）
2. UMLS映射（如果配置了UMLS API key）
3. 现有词典（synonym.json + WordNet作为补充）

# 构建OR查询
query_text = "糖尿病 OR 2型糖尿病 OR T2DM OR type 2 diabetes"
```

#### 4.2.2 并行检索

```python
# 每路检索配置
- 关键词检索：OR连接，minimum_should_match=60%
- 向量检索：维度文本的embedding
- 融合权重：关键词0.7 + 向量0.3
- TopK：每路300-500 chunks
```

#### 4.2.3 文章级筛查

```python
# 必须条件
同一doc_id的chunks中，必须同时命中：
- P维度（至少一个chunk）
- I维度（至少一个chunk）
- O维度（至少一个chunk）

# C处理（可选）
- C存在时，命中C的chunks获得boost
- 但不作为必须条件
```

## 五、集成到现有系统

### 5.1 在API中使用

```python
# api/apps/api_app.py 或 api/apps/sdk/doc.py

from rag.retrieval import PICORetriever
from rag.nlp.search import Dealer
from api.db.services.llm_service import LLMBundle
from api.db import LLMType

@manager.route('/retrieval', methods=['POST'])
def retrieval():
    # ... 现有代码 ...
    
    # 检查是否使用PICO检索
    use_pico = req.get("use_pico", False)
    retrieval_strategy = req.get("retrieval_strategy", "standard")
    
    if use_pico or retrieval_strategy == "pico":
        # 初始化PICO检索器
        embd_mdl = LLMBundle(kbs[0].tenant_id, LLMType.EMBEDDING, llm_name=kbs[0].embd_id)
        chat_mdl = LLMBundle(kbs[0].tenant_id, LLMType.CHAT)
        
        pico_retriever = PICORetriever(
            data_store=settings.retrievaler.dataStore,
            embd_mdl=embd_mdl,
            chat_mdl=chat_mdl,
            strict_mode=req.get("pico_strict_mode", True)
        )
        
        ranks = pico_retriever.retrieve(
            question=question,
            tenant_ids=[kbs[0].tenant_id],
            kb_ids=kb_ids,
            page=page,
            page_size=size,
            topk_per_dimension=300,
            max_chunks_per_doc=10
        )
    else:
        # 使用标准检索
        ranks = settings.retrievaler.retrieval(...)
    
    # ... 返回结果 ...
```

### 5.2 配置参数

```yaml
# config.yaml 或环境变量

retrieval:
  strategies:
    - standard  # 标准RAG检索
    - pico      # PICO框架检索
  
  pico:
    enabled: true
    strict_mode: true  # 严格模式：必须P+I+O
    topk_per_dimension: 300
    max_chunks_per_doc: 10
    
    # UMLS配置（可选）
    umls:
      enabled: false
      api_key: ""  # UMLS API key
      base_url: "https://uts-ws.nlm.nih.gov"
    
    # 重排序权重
    ranking:
      pio_coverage_weight: 0.4
      proximity_weight: 0.2
      section_weight: 0.3
      position_weight: 0.1
```

### 5.3 自动策略选择（可选）

```python
def is_evidence_based_medical_query(question: str) -> bool:
    """
    判断是否为循证医学查询
    可使用简单的关键词匹配或LLM判断
    """
    evidence_keywords = [
        "能否", "是否", "效果", "疗效", "治疗", "干预",
        "患者", "人群", "结局", "结果", "对比", "比较"
    ]
    
    # 简单策略：包含多个关键词
    count = sum(1 for kw in evidence_keywords if kw in question)
    return count >= 2
```

## 六、使用示例

### 6.1 基本使用

```python
from rag.retrieval import PICORetriever
from api.db.services.llm_service import LLMBundle
from api.db import LLMType
from api import settings

# 初始化
embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, llm_name="your_embedding_model")
chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)

pico_retriever = PICORetriever(
    data_store=settings.retrievaler.dataStore,
    embd_mdl=embd_mdl,
    chat_mdl=chat_mdl,
    strict_mode=True
)

# 执行检索
question = "糖尿病患者使用二甲双胍能否降低血糖？"
results = pico_retriever.retrieve(
    question=question,
    tenant_ids=[tenant_id],
    kb_ids=[kb_id],
    page=1,
    page_size=30
)

# 结果包含
# - total: 总结果数
# - chunks: 检索到的chunks列表
# - pico_info: PICO抽取信息（P/I/O/C）
```

### 6.2 带UMLS映射

```python
from rag.retrieval import PICORetriever, UMLSMapper

# 初始化UMLS映射器（需要UMLS API key）
umls_mapper = UMLSMapper(
    api_key="your_umls_api_key",
    base_url="https://uts-ws.nlm.nih.gov"
)

pico_retriever = PICORetriever(
    data_store=settings.retrievaler.dataStore,
    embd_mdl=embd_mdl,
    chat_mdl=chat_mdl,
    umls_mapper=umls_mapper,
    strict_mode=True
)
```

### 6.3 宽松模式

```python
# 宽松模式：允许P+I或I+O组合
pico_retriever = PICORetriever(
    data_store=settings.retrievaler.dataStore,
    embd_mdl=embd_mdl,
    chat_mdl=chat_mdl,
    strict_mode=False  # 宽松模式
)
```

## 七、性能优化建议

### 7.1 缓存策略

```python
# 使用Redis缓存PICO提取结果
from rag.utils.redis_conn import RedisConn

cache = RedisConn().client
pico_retriever = PICORetriever(
    ...,
    cache=cache  # 传入缓存对象
)
```

### 7.2 并行检索优化

```python
# 三路检索已使用ThreadPoolExecutor并行执行
# 如需进一步优化，可调整max_workers
```

### 7.3 TopK控制

```python
# 根据场景调整每路检索的TopK
results = pico_retriever.retrieve(
    ...,
    topk_per_dimension=200,  # 减少检索数量，提高速度
    max_chunks_per_doc=5     # 减少每篇文章的chunks数
)
```

## 八、测试与评估

### 8.1 测试用例

```python
test_queries = [
    "糖尿病患者使用二甲双胍能否降低血糖？",
    "高血压患者服用ACE抑制剂对心功能有何影响？",
    "阿司匹林预防心梗的有效性如何？",
    "儿童疫苗接种的安全性？"  # 可能提取失败，会回退标准检索
]
```

### 8.2 评估指标

- **Recall@10**: 前10个结果中相关结果的比例
- **Precision@10**: 前10个结果的精确度
- **PIO覆盖率**: 结果中同时包含P/I/O的比例
- **响应时间**: 端到端延迟

### 8.3 与标准RAG对比

```python
# A/B测试
standard_results = standard_retriever.retrieve(...)
pico_results = pico_retriever.retrieve(...)

# 对比评估
compare_results(standard_results, pico_results)
```

## 九、故障排查

### 9.1 PICO提取失败

**现象**：检索结果与标准RAG相同

**原因**：
- LLM提取失败
- PICO结构验证不通过

**处理**：
- 检查LLM服务是否正常
- 查看日志中的错误信息
- 确认查询是否为循证医学问题

### 9.2 召回结果为空

**现象**：`total=0`

**原因**：
- 没有文献同时满足P+I+O
- 同义词扩展不足

**处理**：
- 尝试宽松模式（`strict_mode=False`）
- 检查同义词扩展是否生效
- 增加每路检索的TopK

### 9.3 响应时间过长

**原因**：
- 三路并行检索耗时
- 文章级筛查处理大量chunks

**处理**：
- 减少`topk_per_dimension`
- 启用缓存
- 优化文章级筛查算法

## 十、后续改进方向

1. **UMLS深度集成**：实现完整的UMLS API调用
2. **证据等级评估**：集成研究设计类型、样本量等元数据
3. **指南优先策略**：优先返回临床指南和系统综述
4. **知识图谱扩展**：结合GraphRAG进行关联扩展
5. **多模态检索**：支持图表检索

## 十一、参考文档

- [PICO框架医疗检索系统评估](./PICO_RAG_Evaluation.md)
- [RAGFLOW检索系统文档](../guides/dataset/run_retrieval_test.md)
- [UMLS API文档](https://documentation.uts.nlm.nih.gov/rest/home.html)

