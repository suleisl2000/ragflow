#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
PICO框架医疗检索系统
基于循证医学PICO框架的结构化检索，确保证据完整性和检索精度
"""

import logging
import json
import re
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from rag.nlp import search, rag_tokenizer, synonym, query, is_chinese
from rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, FusionExpr
try:
    from api.db.services.llm_service import LLMBundle
    from api.db import LLMType
except ImportError:
    # 处理导入错误（测试环境）
    LLMBundle = None
    LLMType = None


@dataclass
class PICODimension:
    """PICO维度数据结构"""
    text: str
    keywords: List[str]
    synonyms: List[str]
    confidence: float


@dataclass
class PICOStructure:
    """PICO结构"""
    P: Optional[PICODimension] = None  # Population/Patient
    I: Optional[PICODimension] = None  # Intervention
    C: Optional[PICODimension] = None  # Comparison (可选)
    O: Optional[PICODimension] = None  # Outcome
    notes: str = ""
    fallback: bool = False  # 是否为回退模式


class PICOQueryRewriter:
    """PICO查询结构化抽取器"""
    
    PICO_PROMPT_TEMPLATE = """你是医学循证检索助手。请将以下临床问题转换为标准PICO结构。

问题：{question}

要求：
1. 提取P (人群/患者)、I (干预措施)、O (结局指标)
2. C (对照) 可选，如果问题中未明确提及可不填
3. 为每个维度提取关键词和同义词（包括中英文、缩写等）
4. 给出置信度评分（0-1）

输出JSON格式：
{{
    "P": {{"text": "...", "keywords": ["..."], "synonyms": ["..."], "confidence": 0.0-1.0}},
    "I": {{"text": "...", "keywords": ["..."], "synonyms": ["..."], "confidence": 0.0-1.0}},
    "C": {{"text": "...", "keywords": ["..."], "synonyms": ["..."], "confidence": 0.0-1.0}} 或 null,
    "O": {{"text": "...", "keywords": ["..."], "synonyms": ["..."], "confidence": 0.0-1.0}},
    "notes": "提取说明或歧义说明"
}}

示例：
问题：糖尿病患者使用二甲双胍能否降低血糖？
{{
    "P": {{"text": "糖尿病患者", "keywords": ["糖尿病", "2型糖尿病"], "synonyms": ["T2DM", "type 2 diabetes", "非胰岛素依赖型糖尿病"], "confidence": 0.9}},
    "I": {{"text": "二甲双胍", "keywords": ["二甲双胍"], "synonyms": ["metformin", "双胍类", "biguanides"], "confidence": 0.95}},
    "C": null,
    "O": {{"text": "降低血糖", "keywords": ["血糖", "血糖控制"], "synonyms": ["HbA1c", "糖化血红蛋白", "glycemic control", "血糖下降"], "confidence": 0.85}},
    "notes": "明确的问题结构"
}}
"""
    
    def __init__(self, chat_mdl: LLMBundle, cache=None):
        """
        Args:
            chat_mdl: 用于PICO提取的LLM模型
            cache: 缓存对象（可选，用于缓存提取结果）
        """
        self.chat_mdl = chat_mdl
        self.cache = cache
    
    def extract(self, question: str, max_retries: int = 2) -> PICOStructure:
        """
        提取PICO结构，带容错机制
        
        Args:
            question: 用户查询
            max_retries: 最大重试次数
            
        Returns:
            PICOStructure对象
        """
        logging.info(f"[PICO提取] 开始提取，question={question[:100]}..., max_retries={max_retries}")
        
        # 检查缓存
        cache_key = self._get_cache_key(question)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logging.info(f"[PICO提取] 从缓存中获取结果")
                return self._parse_pico_json(cached)
        
        # 尝试提取
        for attempt in range(max_retries + 1):
            try:
                logging.debug(f"[PICO提取] 尝试 {attempt+1}/{max_retries+1}")
                prompt = self.PICO_PROMPT_TEMPLATE.format(question=question)
                # LLMBundle.chat() 需要 system 和 history 作为位置参数
                response = self.chat_mdl.chat(
                    system="",
                    history=[{"role": "user", "content": prompt}],
                    gen_conf={}
                )
                logging.info(f"[PICO提取] LLM响应长度: {len(response)}")
                logging.info(f"[PICO提取] LLM原始响应: {response[:1000]}")
                
                # 解析JSON响应
                try:
                    pico_json = self._extract_json_from_response(response)
                    logging.info(f"[PICO提取] 解析后的JSON: {pico_json}")
                except Exception as e:
                    logging.error(f"[PICO提取] JSON解析失败: {e}, 响应: {response[:500]}")
                    raise
                pico = self._parse_pico_json(pico_json)
                logging.info(f"[PICO提取] 解析完成，P={pico.P.text if pico.P else None}, "
                            f"I={pico.I.text if pico.I else None}, O={pico.O.text if pico.O else None}")
                logging.info(f"[PICO提取] P对象存在={pico.P is not None}, I对象存在={pico.I is not None}, "
                            f"O对象存在={pico.O is not None}")
                
                # 验证PICO结构
                if self._validate_pico(pico):
                    logging.info(f"[PICO提取] 验证通过，P置信度={pico.P.confidence}, "
                               f"I置信度={pico.I.confidence}, O置信度={pico.O.confidence}")
                    # 缓存结果
                    if self.cache:
                        self.cache.set(cache_key, pico_json, ttl=3600)
                        logging.debug(f"[PICO提取] 结果已缓存")
                    return pico
                else:
                    logging.warning(f"[PICO提取] 验证失败，P={pico.P is not None}, I={pico.I is not None}, "
                                  f"O={pico.O is not None}")
                
            except Exception as e:
                logging.warning(f"[PICO提取] 提取失败 (尝试 {attempt+1}/{max_retries+1}): {e}", exc_info=True)
                if attempt == max_retries:
                    break
        
        # 全部失败，返回回退模式
        logging.warning(f"[PICO提取] 所有尝试均失败，返回回退模式: {question}")
        return PICOStructure(fallback=True, notes="PICO提取失败，回退模式")
    
    def _extract_json_from_response(self, response: str) -> dict:
        """从LLM响应中提取JSON"""
        
        # 方法1: 尝试找到第一个 { 和最后一个 }，处理嵌套大括号
        start_idx = response.find('{')
        if start_idx == -1:
            raise ValueError("响应中未找到JSON起始大括号")
        
        # 从第一个 { 开始，计算大括号的嵌套深度来找到匹配的 }
        depth = 0
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                depth += 1
            elif response[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = response[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logging.warning(f"[PICO提取] JSON解析失败: {e}, JSON片段: {json_str[:200]}")
                        break
        
        # 方法2: 尝试使用正则表达式匹配（如果方法1失败）
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 方法3: 尝试提取被```json```或```包围的JSON
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法从响应中提取有效JSON，响应前500字符: {response[:500]}")
    
    def _parse_pico_json(self, pico_json: dict) -> PICOStructure:
        """解析PICO JSON为PICOStructure对象"""
        pico = PICOStructure()
        
        for dim in ['P', 'I', 'C', 'O']:
            if dim in pico_json and pico_json[dim]:
                data = pico_json[dim]
                setattr(pico, dim, PICODimension(
                    text=data.get('text', ''),
                    keywords=data.get('keywords', []),
                    synonyms=data.get('synonyms', []),
                    confidence=data.get('confidence', 0.5)
                ))
        
        pico.notes = pico_json.get('notes', '')
        return pico
    
    def _validate_pico(self, pico: PICOStructure) -> bool:
        """验证PICO结构的完整性"""
        # 必须至少要有P、I、O
        if not pico.P or not pico.I or not pico.O:
            return False
        
        # 检查置信度
        min_confidence = 0.3
        if pico.P.confidence < min_confidence or \
           pico.I.confidence < min_confidence or \
           pico.O.confidence < min_confidence:
            return False
        
        # 检查关键词是否为空
        if not pico.P.keywords or not pico.I.keywords or not pico.O.keywords:
            return False
        
        return True
    
    def _get_cache_key(self, question: str) -> str:
        """生成缓存key"""
        return f"pico:{hashlib.md5(question.encode()).hexdigest()}"


class UMLSMapper:
    """UMLS医学本体映射器（可选，需要UMLS API key）"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://uts-ws.nlm.nih.gov"):
        self.api_key = api_key
        self.base_url = base_url
        self.enabled = api_key is not None
    
    def get_synonyms(self, term: str) -> List[str]:
        """获取UMLS同义词"""
        if not self.enabled:
            return []
        
        try:
            # TODO: 实现UMLS API调用
            # 这里需要UMLS REST API实现
            # 示例：通过UMLS API搜索CUI，然后获取同义词
            pass
        except Exception as e:
            logging.warning(f"UMLS映射失败: {e}")
        
        return []


class PICOSynonymExpander:
    """PICO同义词扩展器"""
    
    def __init__(self, umls_mapper: Optional[UMLSMapper] = None):
        """
        Args:
            umls_mapper: UMLS映射器（可选）
        """
        self.umls_mapper = umls_mapper
        self.syn_dealer = synonym.Dealer()
    
    def expand(self, dimension: PICODimension) -> List[str]:
        """
        扩展PICO维度的同义词
        
        Args:
            dimension: PICO维度对象
            
        Returns:
            扩展后的同义词列表（包含原始关键词）
        """
        logging.debug(f"[同义词扩展] 开始扩展，keywords={dimension.keywords}, synonyms={dimension.synonyms[:5]}")
        all_terms = set()
        
        # 1. 添加原始关键词和LLM提取的同义词
        all_terms.update(dimension.keywords)
        all_terms.update(dimension.synonyms)
        logging.debug(f"[同义词扩展] LLM提取后术语数: {len(all_terms)}")
        
        # 2. UMLS映射（如果可用）
        if self.umls_mapper and self.umls_mapper.enabled:
            umls_count = 0
            for term in dimension.keywords:
                umls_syns = self.umls_mapper.get_synonyms(term)
                all_terms.update(umls_syns)
                umls_count += len(umls_syns)
            logging.debug(f"[同义词扩展] UMLS映射添加术语数: {umls_count}")
        
        # 3. 现有词典扩展（作为补充）
        dict_count = 0
        for term in dimension.keywords:
            dict_syns = self.syn_dealer.lookup(term, topn=5)
            all_terms.update(dict_syns)
            dict_count += len(dict_syns)
        logging.debug(f"[同义词扩展] 词典扩展添加术语数: {dict_count}")
        
        result = list(all_terms)
        logging.info(f"[同义词扩展] 扩展完成，原始术语数: {len(dimension.keywords)}, 扩展后术语数: {len(result)}")
        return result
    
    def build_query_text(self, dimension: PICODimension) -> str:
        """
        构建OR查询文本
        
        Args:
            dimension: PICO维度对象
            
        Returns:
            OR连接的查询文本，如: "糖尿病 OR 2型糖尿病 OR T2DM"
        """
        terms = self.expand(dimension)
        
        # 处理术语（针对中文短语的特殊处理）
        # 问题根源：Infinity的whitespace analyzer对无空格中文文本只返回1个token
        #   - 查询文本 "降低血糖" → whitespace analyzer → ["降低血糖"] (1个token)
        #   - 但文档存储时使用rag_tokenizer.tokenize()分词为: ["降低", "血糖"] (2个独立token)
        #   - 引号短语查询查找token "降低血糖"，索引中不存在 → 返回0结果
        # 解决方案：使用和文档存储时相同的分词器(rag_tokenizer)进行分词
        #   - 如果分词结果为多个token，使用AND查询
        #   - 如果分词结果为1个token，直接使用
        def process_term(term_str):
            """处理单个术语，使用与文档存储时相同的分词器"""
            term_str = term_str.strip()
            if not term_str:
                return None
            
            # 转义特殊字符（使用ragflow的标准方法）
            term_str = query.FulltextQueryer.subSpecialChar(term_str)
            
            # 检查是否为中文短语（使用ragflow的is_chinese方法）
            # is_chinese检查文本中中文字符占比是否>20%
            has_chinese = is_chinese(term_str)
            
            # 如果包含空格，可能是英文短语（如 "glycemic control"），用引号包裹
            if ' ' in term_str:
                return f'"{term_str}"'
            
            # 对于中文短语，使用rag_tokenizer进行分词（与文档存储时一致）
            if has_chinese and len(term_str) > 2:
                # 使用相同的分词器进行分词
                tokenized = rag_tokenizer.tokenize(term_str)
                tokens = tokenized.split()
                
                # 如果分词结果为多个token，使用AND查询
                if len(tokens) > 1:
                    # 每个token用引号包裹，用AND连接
                    quoted_tokens = [f'"{token}"' for token in tokens if token.strip()]
                    if quoted_tokens:
                        return ' AND '.join(quoted_tokens)
                    else:
                        return term_str
                else:
                    # 如果分词结果只有1个token，直接返回（不需要引号）
                    return tokens[0] if tokens else term_str
            else:
                # 单个词或不包含中文的术语，直接返回
                return term_str
        
        processed_terms = []
        for term in terms:
            processed = process_term(term)
            if processed:
                processed_terms.append(processed)
        
        if len(processed_terms) == 1:
            query_text = processed_terms[0]
        else:
            query_text = " OR ".join(processed_terms[:50])  # OR连接，限制最多50个术语
        logging.info(f"[查询文本构建] 维度 {dimension.text if hasattr(dimension, 'text') else 'unknown'}, "
                    f"扩展术语数: {len(terms)}, 构建查询: {query_text[:200]}...")
        return query_text


class PICOParallelRetriever:
    """PICO并行检索器"""
    
    def __init__(self, data_store, embd_mdl: LLMBundle):
        """
        Args:
            data_store: DocStoreConnection实例
            embd_mdl: Embedding模型
        """
        self.data_store = data_store
        self.embd_mdl = embd_mdl
        self.synonym_expander = PICOSynonymExpander()
        self.qryr = query.FulltextQueryer()
    
    def retrieve_dimension(self, 
                          dimension: PICODimension,
                          tenant_ids: List[str],
                          kb_ids: List[str],
                          topk: int = 300,
                          similarity_threshold: float = 0.2,
                          keyword_weight: float = 0.7) -> search.Dealer.SearchResult:
        """
        检索单个PICO维度
        
        Args:
            dimension: PICO维度
            tenant_ids: 租户ID列表
            kb_ids: 知识库ID列表
            topk: 返回Top-K结果
            similarity_threshold: 向量相似度阈值
            keyword_weight: 关键词权重（向量权重 = 1 - keyword_weight）
            
        Returns:
            SearchResult对象
        """
        from rag.nlp.search import index_name
        
        # 构建OR查询文本
        query_text = self.synonym_expander.build_query_text(dimension)
        logging.info(f"[维度检索] 开始检索维度，查询文本长度: {len(query_text)}, topk={topk}")
        
        # 构建关键词检索表达式
        # 使用显式OR语法：'term1 OR term2 OR ...'
        # Infinity的match_text支持显式OR语法（默认operator_option=kInfinitySyntax）
        # minimum_should_match="1" 表示至少匹配1个术语，确保OR语义
        match_text = MatchTextExpr(
            fields=["content_ltks", "important_kwd", "title_tks", "question_tks"],
            matching_text=query_text,
            topn=topk,
            extra_options={
                "minimum_should_match": "1"  # 至少匹配1个，确保OR语义
            }
        )
        
        # 构建向量检索表达式
        # 使用维度文本生成embedding
        dim_text = " ".join(dimension.keywords + dimension.synonyms[:5])
        qv, _ = self.embd_mdl.encode_queries(dim_text)
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        
        match_dense = MatchDenseExpr(
            vector_column_name=vector_column_name,
            embedding_data=embedding_data,
            embedding_data_type='float',
            distance_type='cosine',
            topn=topk,
            extra_options={"similarity": similarity_threshold}
        )
        
        # 融合表达式
        vector_weight = 1.0 - keyword_weight
        fusion = FusionExpr(
            "weighted_sum",
            topk,
            {"weights": f"{keyword_weight},{vector_weight}"}
        )
        
        # 执行检索
        idx_names = [index_name(tid) for tid in tenant_ids]
        # 不需要在 condition 中传递 kb_ids，因为：
        # 1. kb_ids 已通过 knowledgebaseIds 参数传递
        # 2. Infinity 通过表名隐式过滤 kb_id
        # 3. ES/OS 会在 search 方法中自动添加 kb_id 到 condition
        filters = {}
        
        logging.info(f"[维度检索执行] 维度 {dimension.text if hasattr(dimension, 'text') else 'unknown'}, "
                    f"查询文本: {query_text[:150]}, fields: {match_text.fields}, "
                    f"extra_options: {match_text.extra_options}")
        
        res, total = self.data_store.search(
            selectFields=["id", "doc_id", "content_ltks", "content_with_weight", vector_column_name, 
                         "title_tks", "important_kwd", "docnm_kwd"],
            highlightFields=[],
            condition=filters,
            matchExprs=[match_text, match_dense, fusion],
            orderBy=None,
            offset=0,
            limit=topk,
            indexNames=idx_names,
            knowledgebaseIds=kb_ids
        )
        
        logging.info(f"[维度检索结果] 维度 {dimension.text if hasattr(dimension, 'text') else 'unknown'}, "
                    f"Infinity返回: total={total}, res.empty={res.empty if hasattr(res, 'empty') else 'N/A'}, "
                    f"res.shape={res.shape if hasattr(res, 'shape') else 'N/A'}")
        
        # 转换为SearchResult格式
        # 使用dataStore.getFields方法处理字段格式转换（处理keyword字段、列表等）
        # Infinity 表中的字段名是 "id"，不是 "chunk_id"
        select_fields = ["id", "doc_id", "content_ltks", "content_with_weight", vector_column_name, 
                        "title_tks", "important_kwd", "docnm_kwd"]
        fields = self.data_store.getFields(res, select_fields)
        chunk_ids = self.data_store.getChunkIds(res)
        
        # 输出匹配的chunk ID和内容（用于调试和对比）
        logging.info(f"[维度检索结果] 维度 {dimension.text if hasattr(dimension, 'text') else 'unknown'}, "
                    f"匹配到 {len(chunk_ids)} 个chunks")
        if chunk_ids:
            logging.info(f"[维度检索结果] 匹配的chunk IDs: {chunk_ids[:20]}{'...' if len(chunk_ids) > 20 else ''}")
            # 输出前10个chunks的详细信息
            for i, chunk_id in enumerate(chunk_ids[:10], 1):
                chunk_info = fields.get(chunk_id, {})
                doc_id = chunk_info.get('doc_id', '')
                doc_name = chunk_info.get('docnm_kwd', '')
                content = chunk_info.get('content_with_weight', '') or chunk_info.get('content_ltks', '')
                content_preview = content[:200] + "..." if len(content) > 200 else content
                logging.info(f"[维度检索结果]   Chunk {i} (ID: {chunk_id}, Doc: {doc_name}):")
                logging.info(f"[维度检索结果]     Content: {content_preview}")
        
        return search.Dealer.SearchResult(
            total=total,
            ids=chunk_ids,
            query_vector=embedding_data,
            field=fields,
            highlight=None,
            aggregation=None,
            keywords=dimension.keywords + dimension.synonyms[:10]
        )
    
    def parallel_retrieve(self,
                         pico: PICOStructure,
                         tenant_ids: List[str],
                         kb_ids: List[str],
                         topk_per_dimension: int = 300) -> Dict[str, search.Dealer.SearchResult]:
        """
        并行检索P/I/O三个维度
        
        Args:
            pico: PICO结构
            tenant_ids: 租户ID列表
            kb_ids: 知识库ID列表
            topk_per_dimension: 每个维度返回Top-K
            
        Returns:
            Dict[str, SearchResult]: {"P": ..., "I": ..., "O": ..., "C": ...}
        """
        logging.info(f"[并行检索] 开始并行检索，维度: P={pico.P is not None}, I={pico.I is not None}, "
                    f"O={pico.O is not None}, C={pico.C is not None}, topk_per_dimension={topk_per_dimension}")
        
        results = {}
        
        # 并行检索P/I/O（必须）
        import concurrent.futures
        
        dimensions_to_retrieve = []
        if pico.P:
            dimensions_to_retrieve.append(("P", pico.P))
        if pico.I:
            dimensions_to_retrieve.append(("I", pico.I))
        if pico.O:
            dimensions_to_retrieve.append(("O", pico.O))
        if pico.C:  # C可选，但如果有也检索
            dimensions_to_retrieve.append(("C", pico.C))
        
        logging.info(f"[并行检索] 准备检索 {len(dimensions_to_retrieve)} 个维度")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.retrieve_dimension,
                    dim,
                    tenant_ids,
                    kb_ids,
                    topk_per_dimension
                ): name
                for name, dim in dimensions_to_retrieve
            }
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                    logging.info(f"[并行检索] 维度 {name} 检索完成，返回 {len(result.ids)} 个chunks")
                except Exception as e:
                    logging.error(f"[并行检索] 维度 {name} 检索失败: {e}", exc_info=True)
                    results[name] = search.Dealer.SearchResult(
                        total=0, ids=[], field={}, keywords=[]
                    )
        
        # 安全地获取结果，确保都是 SearchResult 对象
        p_result = results.get('P')
        i_result = results.get('I')
        o_result = results.get('O')
        c_result = results.get('C')
        
        p_count = len(p_result.ids) if isinstance(p_result, search.Dealer.SearchResult) else 0
        i_count = len(i_result.ids) if isinstance(i_result, search.Dealer.SearchResult) else 0
        o_count = len(o_result.ids) if isinstance(o_result, search.Dealer.SearchResult) else 0
        c_count = len(c_result.ids) if isinstance(c_result, search.Dealer.SearchResult) else 0
        
        logging.info(f"[并行检索] 所有维度检索完成，P={p_count}, I={i_count}, O={o_count}, C={c_count}")
        return results


class PIOArticleFilter:
    """文章级PIO一致性筛查器"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: 严格模式（必须同时满足P+I+O）
        """
        self.strict_mode = strict_mode
    
    def filter(self,
               pico_results: Dict[str, search.Dealer.SearchResult]) -> Dict[str, List[str]]:
        """
        筛选满足PIO条件的文章
        
        Args:
            pico_results: 各维度检索结果 {"P": SearchResult, "I": ..., "O": ...}
            
        Returns:
            Dict[str, List[str]]: {doc_id: [chunk_ids]} - 满足条件的文章及其chunks
        """
        logging.info(f"[文章级筛查] 开始筛查，strict_mode={self.strict_mode}")
        
        if 'P' not in pico_results or 'I' not in pico_results or 'O' not in pico_results:
            logging.warning(f"[文章级筛查] P/I/O维度不完整，无法进行文章级筛查，可用维度: {list(pico_results.keys())}")
            return {}
        
        # 统计每个chunk的P/I/O命中情况
        chunk_hits = defaultdict(lambda: {"P": False, "I": False, "O": False, "C": False})
        doc_chunks = defaultdict(set)  # 使用set避免重复
        
        for dim_name, search_result in pico_results.items():
            for chunk_id in search_result.ids:
                # 获取doc_id
                field = search_result.field.get(chunk_id, {})
                doc_id = field.get('doc_id', '')
                if not doc_id:
                    continue
                
                doc_chunks[doc_id].add(chunk_id)  # 使用add而不是append，自动去重
                chunk_hits[chunk_id][dim_name] = True
        
        # 转换为list以便后续处理
        doc_chunks = {doc_id: list(chunk_ids) for doc_id, chunk_ids in doc_chunks.items()}
        
        logging.debug(f"[文章级筛查] 涉及 {len(doc_chunks)} 篇文章，总计 {sum(len(cids) for cids in doc_chunks.values())} 个chunks")
        
        # 文章级筛查：检查每篇文章是否同时包含P/I/O的chunks
        valid_docs = {}
        
        for doc_id, chunk_ids in doc_chunks.items():
            # 找出命中每个维度的chunk ID列表，用于详细日志
            # 使用list(dict.fromkeys())去重，保持顺序
            p_hit_chunks = list(dict.fromkeys([cid for cid in chunk_ids if chunk_hits[cid]["P"]]))
            i_hit_chunks = list(dict.fromkeys([cid for cid in chunk_ids if chunk_hits[cid]["I"]]))
            o_hit_chunks = list(dict.fromkeys([cid for cid in chunk_ids if chunk_hits[cid]["O"]]))
            
            doc_p_hit = len(p_hit_chunks) > 0
            doc_i_hit = len(i_hit_chunks) > 0
            doc_o_hit = len(o_hit_chunks) > 0
            
            # 严格模式：必须同时满足P+I+O
            if self.strict_mode:
                if doc_p_hit and doc_i_hit and doc_o_hit:
                    valid_docs[doc_id] = chunk_ids
                    # 获取文档标题用于日志
                    doc_title = None
                    for chunk_id in chunk_ids:
                        for search_result in pico_results.values():
                            if chunk_id in search_result.ids:
                                field = search_result.field.get(chunk_id, {})
                                doc_title = field.get('docnm_kwd', '')
                                if doc_title:
                                    break
                        if doc_title:
                            break
                    logging.info(f"[文章级筛查] 文章通过筛查（严格模式）: doc_id={doc_id}, doc_title={doc_title}, "
                               f"P命中{len(p_hit_chunks)}个chunks, I命中{len(i_hit_chunks)}个chunks, "
                               f"O命中{len(o_hit_chunks)}个chunks")
                    
                    # 输出每个维度命中的chunk内容示例（前2个）
                    if p_hit_chunks:
                        logging.info(f"[文章级筛查]   P维度命中chunk示例:")
                        for i, cid in enumerate(p_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
                    if i_hit_chunks:
                        logging.info(f"[文章级筛查]   I维度命中chunk示例:")
                        for i, cid in enumerate(i_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
                    if o_hit_chunks:
                        logging.info(f"[文章级筛查]   O维度命中chunk示例:")
                        for i, cid in enumerate(o_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
            else:
                # 宽松模式：至少满足P+I或I+O
                if (doc_p_hit and doc_i_hit) or (doc_i_hit and doc_o_hit):
                    valid_docs[doc_id] = chunk_ids
                    # 获取文档标题用于日志
                    doc_title = None
                    for chunk_id in chunk_ids:
                        for search_result in pico_results.values():
                            if chunk_id in search_result.ids:
                                field = search_result.field.get(chunk_id, {})
                                doc_title = field.get('docnm_kwd', '')
                                if doc_title:
                                    break
                        if doc_title:
                            break
                    logging.info(f"[文章级筛查] 文章通过筛查（宽松模式）: doc_id={doc_id}, doc_title={doc_title}, "
                               f"P命中{len(p_hit_chunks)}个chunks, I命中{len(i_hit_chunks)}个chunks, "
                               f"O命中{len(o_hit_chunks)}个chunks")
                    
                    # 输出每个维度命中的chunk内容示例（前2个）
                    if p_hit_chunks:
                        logging.info(f"[文章级筛查]   P维度命中chunk示例:")
                        for i, cid in enumerate(p_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
                    if i_hit_chunks:
                        logging.info(f"[文章级筛查]   I维度命中chunk示例:")
                        for i, cid in enumerate(i_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
                    if o_hit_chunks:
                        logging.info(f"[文章级筛查]   O维度命中chunk示例:")
                        for i, cid in enumerate(o_hit_chunks[:2]):
                            for search_result in pico_results.values():
                                if cid in search_result.ids:
                                    field = search_result.field.get(cid, {})
                                    content = field.get('content_with_weight', '') or field.get('content_ltks', '')
                                    if content:
                                        logging.info(f"[文章级筛查]     Chunk {cid}: {content[:150]}...")
                                        break
        
        logging.info(f"[文章级筛查] 筛查完成，通过 {len(valid_docs)}/{len(doc_chunks)} 篇文章")
        return valid_docs


class IntraDocChunkRanker:
    """文章内chunk重排序器"""
    
    def __init__(self,
                 pio_coverage_weight: float = 0.4,
                 proximity_weight: float = 0.2,
                 section_weight: float = 0.3,
                 position_weight: float = 0.1):
        """
        Args:
            pio_coverage_weight: PIO覆盖密度权重
            proximity_weight: 邻近性权重
            section_weight: 章节权重
            position_weight: 位置权重
        """
        self.pio_coverage_weight = pio_coverage_weight
        self.proximity_weight = proximity_weight
        self.section_weight = section_weight
        self.position_weight = position_weight
        
        # 章节优先级（数值越大优先级越高）
        self.section_priority = {
            "results": 1.0,
            "conclusion": 0.9,
            "methods": 0.8,
            "abstract": 0.6,
            "introduction": 0.5,
            "discussion": 0.7,
        }
    
    def rank(self,
             doc_id: str,
             chunk_ids: List[str],
             pico_results: Dict[str, search.Dealer.SearchResult],
             max_chunks_per_doc: int = 10) -> List[str]:
        """
        对文章内的chunks进行重排序
        
        Args:
            doc_id: 文章ID
            chunk_ids: 文章内的chunk IDs
            pico_results: 各维度检索结果
            max_chunks_per_doc: 每篇文章最多返回的chunks数
            
        Returns:
            排序后的chunk IDs
        """
        if not chunk_ids:
            return []
        
        # 计算每个chunk的PIO覆盖度
        chunk_hits = {}
        for chunk_id in chunk_ids:
            hits = {
                "P": any(chunk_id in res.ids for res in [pico_results.get("P")] if res),
                "I": any(chunk_id in res.ids for res in [pico_results.get("I")] if res),
                "O": any(chunk_id in res.ids for res in [pico_results.get("O")] if res),
                "C": any(chunk_id in res.ids for res in [pico_results.get("C")] if res),
            }
            chunk_hits[chunk_id] = hits
        
        # 计算每个chunk的评分
        chunk_scores = {}
        
        for chunk_id in chunk_ids:
            score = 0.0
            
            # 1. PIO覆盖密度（同时命中多个维度加分）
            pio_count = sum([chunk_hits[chunk_id][dim] for dim in ["P", "I", "O"]])
            pio_score = pio_count / 3.0  # 归一化到[0, 1]
            score += self.pio_coverage_weight * pio_score
            
            # 2. 章节权重（从field中获取章节信息，如果有）
            # 这里简化处理，实际可以从chunk的metadata中获取section信息
            section_score = 0.5  # 默认值
            # TODO: 从chunk metadata中提取section，使用self.section_priority
            score += self.section_weight * section_score
            
            # 3. 位置权重（前面的chunks优先级稍高）
            position_idx = chunk_ids.index(chunk_id)
            position_score = 1.0 - (position_idx / len(chunk_ids)) * 0.3  # 最多降低30%
            score += self.position_weight * position_score
            
            # 4. C boost（如果命中C，额外加分）
            if chunk_hits[chunk_id]["C"]:
                score += 0.1
            
            chunk_scores[chunk_id] = score
        
        # 按分数排序
        sorted_chunks = sorted(chunk_ids, key=lambda cid: chunk_scores.get(cid, 0), reverse=True)
        
        return sorted_chunks[:max_chunks_per_doc]


class PICORetriever:
    """PICO检索主类"""
    
    def __init__(self,
                 data_store,
                 embd_mdl: LLMBundle,
                 chat_mdl: LLMBundle,
                 umls_mapper: Optional[UMLSMapper] = None,
                 strict_mode: bool = True,
                 cache=None):
        """
        Args:
            data_store: DocStoreConnection实例
            embd_mdl: Embedding模型
            chat_mdl: 用于PICO提取的Chat模型
            umls_mapper: UMLS映射器（可选）
            strict_mode: 严格模式（必须P+I+O）
            cache: 缓存对象
        """
        self.data_store = data_store
        self.embd_mdl = embd_mdl
        self.chat_mdl = chat_mdl
        
        self.rewriter = PICOQueryRewriter(chat_mdl, cache)
        self.retriever = PICOParallelRetriever(data_store, embd_mdl)
        self.filter = PIOArticleFilter(strict_mode)
        self.ranker = IntraDocChunkRanker()
    
    def retrieval(self,
                  question: str,
                  tenant_ids: List[str],
                  kb_ids: List[str],
                  page: int = 1,
                  page_size: int = 30,
                  topk_per_dimension: int = 300,
                  max_chunks_per_doc: int = 10) -> dict:
        """
        执行PICO检索
        
        Args:
            question: 用户查询
            tenant_ids: 租户ID列表
            kb_ids: 知识库ID列表
            page: 页码
            page_size: 每页大小
            topk_per_dimension: 每个维度检索Top-K
            max_chunks_per_doc: 每篇文章最多返回chunks数
            
        Returns:
            检索结果字典，格式与标准检索一致
        """
        # 1. PICO提取
        logging.info(f"[PICO检索] 步骤1: 开始PICO提取，question={question[:100]}...")
        pico = self.rewriter.extract(question)
        logging.info(f"[PICO检索] 步骤1: PICO提取完成，fallback={pico.fallback}, P={pico.P.text if pico.P else None}, I={pico.I.text if pico.I else None}, O={pico.O.text if pico.O else None}")
        
        # 如果提取失败，抛出异常让上层处理
        if pico.fallback:
            logging.warning(f"[PICO检索] PICO提取失败，将抛出异常让上层回退到标准检索")
            raise Exception(f"PICO提取失败，无法进行PICO检索: {pico.notes}")
        
        # 2. 并行检索P/I/O
        logging.info(f"[PICO检索] 步骤2: 开始并行检索P/I/O，topk_per_dimension={topk_per_dimension}")
        pico_results = self.retriever.parallel_retrieve(
            pico, tenant_ids, kb_ids, topk_per_dimension
        )
        logging.info(f"[PICO检索] 步骤2: 并行检索完成，P维度: {len(pico_results.get('P', {}).ids)} chunks, "
                    f"I维度: {len(pico_results.get('I', {}).ids)} chunks, "
                    f"O维度: {len(pico_results.get('O', {}).ids)} chunks")
        
        # 输出PICO提取的关键词详情（用于调试为什么某些文档会匹配）
        logging.info(f"[PICO检索] PICO提取的关键词详情:")
        logging.info(f"[PICO检索]   P维度: text={pico.P.text if pico.P else None}, "
                    f"keywords={pico.P.keywords if pico.P else []}, "
                    f"synonyms={pico.P.synonyms[:10] if pico.P and pico.P.synonyms else []}")
        logging.info(f"[PICO检索]   I维度: text={pico.I.text if pico.I else None}, "
                    f"keywords={pico.I.keywords if pico.I else []}, "
                    f"synonyms={pico.I.synonyms[:10] if pico.I and pico.I.synonyms else []}")
        logging.info(f"[PICO检索]   O维度: text={pico.O.text if pico.O else None}, "
                    f"keywords={pico.O.keywords if pico.O else []}, "
                    f"synonyms={pico.O.synonyms[:10] if pico.O and pico.O.synonyms else []}")
        
        # 3. 文章级PIO筛查
        logging.info(f"[PICO检索] 步骤3: 开始文章级PIO筛查，strict_mode={self.filter.strict_mode}")
        valid_docs = self.filter.filter(pico_results)
        logging.info(f"[PICO检索] 步骤3: 文章级筛查完成，满足条件的文章数: {len(valid_docs)}")
        
        if not valid_docs:
            logging.warning(f"[PICO检索] 未找到满足PIO条件的文章，返回空结果")
            return {"total": 0, "chunks": [], "doc_aggs": {}}
        
        # 输出满足PIO条件的文章标题和chunks内容
        logging.info(f"[PICO检索] ========== 满足PIO条件的文章详情 ==========")
        for doc_id, chunk_ids in valid_docs.items():
            # 获取文档标题（从任意一个chunk中获取docnm_kwd）
            doc_title = None
            doc_chunks_info = []
            
            for chunk_id in chunk_ids:
                chunk_info = None
                for search_result in pico_results.values():
                    if chunk_id in search_result.ids:
                        chunk_info = search_result.field.get(chunk_id, {})
                        break
                
                if chunk_info:
                    if not doc_title:
                        doc_title = chunk_info.get("docnm_kwd", "")
                    # 获取chunk内容
                    content = chunk_info.get("content_with_weight", "") or chunk_info.get("content_ltks", "")
                    if content:
                        doc_chunks_info.append({
                            "chunk_id": chunk_id,
                            "content_preview": content[:200] + "..." if len(content) > 200 else content
                        })
            
            logging.info(f"[PICO检索] 文章ID: {doc_id}")
            logging.info(f"[PICO检索] 文章标题: {doc_title}")
            logging.info(f"[PICO检索] 包含chunks数: {len(chunk_ids)}")
            for idx, chunk_info in enumerate(doc_chunks_info[:8], 1):  # 只显示前5个chunks
                logging.info(f"[PICO检索]   Chunk {idx} (ID: {chunk_info['chunk_id']}):")
                logging.info(f"[PICO检索]     {chunk_info['content_preview']}")
            if len(doc_chunks_info) > 8:
                logging.info(f"[PICO检索]   ... 还有 {len(doc_chunks_info) - 5} 个chunks未显示")
        logging.info(f"[PICO检索] ============================================")
        
        # 4. 文章内chunk重排序
        logging.info(f"[PICO检索] 步骤4: 开始文章内chunk重排序，max_chunks_per_doc={max_chunks_per_doc}")
        all_chunks = []
        docs_to_process = list(valid_docs.items())[:(page * page_size)]
        logging.info(f"[PICO检索] 步骤4: 处理 {len(docs_to_process)} 篇文章")
        
        for doc_id, chunk_ids in docs_to_process:
            ranked_chunks = self.ranker.rank(
                doc_id, chunk_ids, pico_results, max_chunks_per_doc
            )
            logging.debug(f"[PICO检索] 步骤4: 文章 {doc_id} 重排序完成，保留 {len(ranked_chunks)} 个chunks")
            all_chunks.extend(ranked_chunks)
        
        logging.info(f"[PICO检索] 步骤4: 重排序完成，总计 {len(all_chunks)} 个chunks")
        
        # 5. 构建返回结果（格式与标准检索一致）
        logging.info(f"[PICO检索] 步骤5: 开始构建返回结果")
        chunks = []
        for chunk_id in all_chunks:
            # 从任意一个SearchResult中获取chunk信息
            chunk_info = None
            for search_result in pico_results.values():
                if chunk_id in search_result.ids:
                    chunk_info = search_result.field.get(chunk_id, {})
                    break
            
            if chunk_info:
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": chunk_info.get("doc_id", ""),
                    "content_with_weight": chunk_info.get("content_with_weight", "") or chunk_info.get("content_ltks", ""),
                    "similarity": 1.0,  # TODO: 计算实际相似度
                    "docnm_kwd": chunk_info.get("docnm_kwd", ""),
                })
        
        # 分页
        total = len(chunks)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        chunks = chunks[start_idx:end_idx]
        
        result = {
            "total": total,
            "chunks": chunks,
            "doc_aggs": {},
            "pico_info": {
                "P": pico.P.text if pico.P else None,
                "I": pico.I.text if pico.I else None,
                "O": pico.O.text if pico.O else None,
                "C": pico.C.text if pico.C else None,
            }
        }
        
        logging.info(f"[PICO检索] 步骤5: 结果构建完成，total={total}, 当前页chunks数={len(chunks)}, "
                    f"page={page}, page_size={page_size}")
        logging.info(f"[PICO检索] PICO信息: P={result['pico_info']['P']}, I={result['pico_info']['I']}, "
                    f"O={result['pico_info']['O']}, C={result['pico_info']['C']}")
        
        return result

