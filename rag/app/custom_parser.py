#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

import logging
import re
import json
import os
import copy
import hashlib
import requests
from typing import List, Dict, Any, Optional
from collections import Counter

from rag.nlp import rag_tokenizer, tokenize, add_positions, tokenize_chunks, bullets_category, title_frequency
from api.db import ParserType, LLMType
from api.db.services.llm_service import LLMBundle         
from rag.prompts.generator import keyword_extraction
from rag.utils.storage_factory import STORAGE_IMPL
from api.utils.configs import read_config

logger = logging.getLogger(__name__)


class OCRClient:
    """
    Textin OCR API客户端
    """
    def __init__(self, app_id: str, secret_code: str, api_url: str = None):
        self.app_id = app_id
        self.secret_code = secret_code
        self.api_url = api_url or "https://api.textin.com/ai/service/v1/pdf_to_markdown"
    
    def recognize(self, file_content: bytes, options: dict = None) -> str:
        """
        调用OCR API解析PDF文件
        
        Args:
            file_content: PDF文件的二进制内容
            options: OCR选项（可选）
        
        Returns:
            OCR API返回的JSON字符串（包含result字段）
        """
        if options is None:
            options = {}
        
        # 构建请求参数
        params = {}
        for key, value in options.items():
            params[key] = str(value)
        
        # 设置请求头
        headers = {
            "x-ti-app-id": self.app_id,
            "x-ti-secret-code": self.secret_code,
            "Content-Type": "application/octet-stream"
        }
        
        # 发送请求
        response = requests.post(
            self.api_url,
            params=params,
            headers=headers,
            data=file_content,
            timeout=300  # OCR可能需要较长时间
        )
        
        # 检查响应状态
        response.raise_for_status()
        return response.text


class CustomPdfParser:
    """
    自定义解析器 - 支持JSON格式文件解析
    基于Textin JSON格式和章节标题提取
    """
    
    def __init__(self, **kwargs):
        # 设置解析器类型标识
        self.model_speciess = ParserType.CUSTOM
        
        # 自定义配置参数
        self.custom_config = kwargs.get("custom_config", {})
        self.enable_smart_chunking = self.custom_config.get("enable_smart_chunking", True)
        self.custom_delimiter = self.custom_config.get("delimiter", "\n!?;。；！？")
        
        # 章节标题层级管理
        self.title_hierarchy: List[str] = []
        self.current_hierarchy: List[str] = []
        
        # LLM模型初始化（用于关键词生成）
        self.chat_mdl = None
        self._init_llm_model()
        
        # 关键词缓存（基于内容hash）
        self._keyword_cache = {}
        self._max_cache_size = 100  # 最大缓存条目数
        
        # OCR配置初始化
        self.ocr_client = None
        self._init_ocr_config()
        
        # 缓存bucket配置（MinIO bucket名称不能包含下划线，使用连字符）
        self.pdf_cache_bucket = self.custom_config.get("pdf_cache_bucket", "pdf-cache")
        self.json_cache_bucket = self.custom_config.get("json_cache_bucket", "json-cache")
        logger.info(f"[解析器初始化] 缓存bucket配置: PDF={self.pdf_cache_bucket}, JSON={self.json_cache_bucket}")
    
    def _init_llm_model(self):
        """初始化LLM模型用于关键词生成"""
        try:

            # 获取租户ID
            tenant_id = self.custom_config.get("tenant_id")
            if tenant_id:
                # 直接使用LLMBundle初始化，不指定具体模型名称，使用默认模型
                self.chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
                #self.chat_mdl = None
                logger.info(f"Custom parser LLM model initialized for tenant: {tenant_id}")
            else:
                logger.warning("No tenant_id provided, keyword generation will be disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM model for keyword generation: {e}")
            self.chat_mdl = None
    
    def _generate_keywords(self, content: str, topn: int = 5, context: str = "") -> tuple:
        """
        统一的关键词生成方法（带缓存）
        
        Args:
            content: 要生成关键词的内容
            topn: 生成关键词的数量
            context: 上下文信息（用于日志）
        
        Returns:
            tuple: (important_kwd_list, important_tks_string)
        """
        # 使用内容hash作为缓存key
        cache_key = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # 检查缓存
        if cache_key in self._keyword_cache:
            logger.info(f"Cache hit for keywords: {context}")
            print(f"Cache hit for keywords: {context}")
            return self._keyword_cache[cache_key]
        
        # 生成关键词
        result = self._generate_keywords_impl(content, topn, context)
        
        # 缓存结果
        self._keyword_cache[cache_key] = result
        
        # 限制缓存大小
        if len(self._keyword_cache) > self._max_cache_size:
            # 清理最旧的缓存项（FIFO策略）
            oldest_key = next(iter(self._keyword_cache))
            del self._keyword_cache[oldest_key]
            logger.debug(f"Cache size limit reached, removed oldest entry: {oldest_key}")
        
        return result
    
    def _generate_keywords_impl(self, content: str, topn: int = 5, context: str = "") -> tuple:
        """
        实际的关键词生成实现
        """
        if self.chat_mdl:
            try:
                logger.debug(f"[关键词生成] 使用LLM生成关键词: context={context}, topn={topn}, 内容长度={len(content)}")
                # 使用LLM生成关键词
                generated_keywords = keyword_extraction(self.chat_mdl, content, topn=topn)
                if generated_keywords:
                    # 将生成的关键词按逗号分割并添加到列表中
                    keyword_list = [kw.strip() for kw in generated_keywords.split(",") if kw.strip()]
                    important_tks = rag_tokenizer.fine_grained_tokenize(" ".join(keyword_list))
                    logger.info(f"[关键词生成] ✓ LLM生成关键词成功: context={context}, keywords={keyword_list}")
                    return keyword_list, important_tks
                else:
                    logger.warning(f"[关键词生成] LLM返回空关键词: context={context}，回退到分词方法")
            except Exception as e:
                logger.warning(f"[关键词生成] LLM生成关键词失败: context={context}, 错误: {e}，回退到分词方法", exc_info=True)
        else:
            logger.debug(f"[关键词生成] LLM模型未初始化，使用分词方法: context={context}")
        
        # 回退到原来的分词方法
        tokenized_text = rag_tokenizer.tokenize(content)
        important_tks = rag_tokenizer.fine_grained_tokenize(tokenized_text)
        logger.debug(f"[关键词生成] 使用分词方法生成关键词: context={context}, tokenized_text={tokenized_text[:50]}...")
        return [tokenized_text], important_tks
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            dict: 包含缓存大小、命中率等统计信息
        """
        return {
            "cache_size": len(self._keyword_cache),
            "max_cache_size": self._max_cache_size,
            "cache_usage_percent": (len(self._keyword_cache) / self._max_cache_size) * 100
        }
    
    def clear_cache(self):
        """
        清空关键词缓存
        """
        self._keyword_cache.clear()
        logger.info("Keyword cache cleared")
    
    def _init_ocr_config(self):
        """初始化OCR配置，优先级：环境变量 > 配置文件 > custom_config"""
        try:
            # 从环境变量读取
            app_id = os.getenv('TEXTIN_OCR_APP_ID')
            secret_code = os.getenv('TEXTIN_OCR_SECRET_CODE')
            
            # 如果环境变量没有，从配置文件读取
            if not app_id or not secret_code:
                try:
                    config = read_config('service_conf.yaml')
                    ocr_config = config.get('ocr', {})
                    app_id = app_id or ocr_config.get('app_id')
                    secret_code = secret_code or ocr_config.get('secret_code')
                except Exception as e:
                    logger.debug(f"Failed to read OCR config from service_conf.yaml: {e}")
            
            # 如果还没有，从custom_config读取（向后兼容）
            if not app_id:
                app_id = self.custom_config.get('ocr_app_id')
            if not secret_code:
                secret_code = self.custom_config.get('ocr_secret_code')
            
            # 获取API URL
            api_url = self.custom_config.get('ocr_api_url', "https://api.textin.com/ai/service/v1/pdf_to_markdown")
            
            # 如果配置了app_id和secret_code，创建OCR客户端
            if app_id and secret_code:
                self.ocr_client = OCRClient(app_id, secret_code, api_url)
                logger.info("OCR client initialized successfully")
            else:
                logger.warning("OCR app_id or secret_code not configured, PDF parsing will be disabled")
                self.ocr_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize OCR config: {e}")
            self.ocr_client = None
    
    def _calculate_pdf_md5(self, binary: bytes) -> str:
        """计算PDF的MD5值"""
        return hashlib.md5(binary).hexdigest()
    
    def _get_cached_json(self, md5: str) -> Optional[bytes]:
        """从minio获取缓存的JSON"""
        try:
            cache_key = f"{md5}.json"
            logger.debug(f"[缓存] 检查JSON缓存: bucket={self.json_cache_bucket}, key={cache_key}")
            
            if STORAGE_IMPL.obj_exist(self.json_cache_bucket, cache_key):
                json_binary = STORAGE_IMPL.get(self.json_cache_bucket, cache_key)
                logger.info(f"[缓存] ✓ JSON缓存命中: MD5={md5}, JSON大小={len(json_binary)} bytes")
                return json_binary
            else:
                logger.debug(f"[缓存] ✗ JSON缓存未命中: MD5={md5}")
        except Exception as e:
            logger.warning(f"[缓存] 获取JSON缓存失败: MD5={md5}, 错误: {e}", exc_info=True)
        return None
    
    def _save_to_cache(self, md5: str, pdf_binary: bytes, json_binary: bytes):
        """保存PDF和JSON到minio（分开bucket）"""
        try:
            # 保存PDF到pdf-cache bucket
            pdf_key = f"{md5}.pdf"
            logger.debug(f"[缓存] 保存PDF到缓存: bucket={self.pdf_cache_bucket}, key={pdf_key}, 大小={len(pdf_binary)} bytes")
            STORAGE_IMPL.put(self.pdf_cache_bucket, pdf_key, pdf_binary)
            logger.info(f"[缓存] ✓ PDF保存成功: {self.pdf_cache_bucket}/{pdf_key}")
            
            # 保存JSON到json-cache bucket
            json_key = f"{md5}.json"
            logger.debug(f"[缓存] 保存JSON到缓存: bucket={self.json_cache_bucket}, key={json_key}, 大小={len(json_binary)} bytes")
            STORAGE_IMPL.put(self.json_cache_bucket, json_key, json_binary)
            logger.info(f"[缓存] ✓ JSON保存成功: {self.json_cache_bucket}/{json_key}")
        except Exception as e:
            logger.error(f"[缓存] 保存缓存失败: MD5={md5}, 错误: {e}", exc_info=True)
            # 不抛出异常，允许继续处理
    
    def _call_ocr_api(self, pdf_binary: bytes) -> bytes:
        """调用OCR API解析PDF"""
        if not self.ocr_client:
            raise ValueError("OCR client not initialized. Please configure TEXTIN_OCR_APP_ID and TEXTIN_OCR_SECRET_CODE")
        
        logger.info(f"[OCR API] 调用OCR API解析PDF，文件大小: {len(pdf_binary)} bytes, API URL: {self.ocr_client.api_url}")
        import time
        ocr_start = time.time()
        
        try:
            response_text = self.ocr_client.recognize(pdf_binary)
            ocr_duration = time.time() - ocr_start
            logger.info(f"[OCR API] OCR API调用成功，耗时: {ocr_duration:.2f}秒，响应大小: {len(response_text)} bytes")
            
            # 解析响应，提取result字段
            json_response = json.loads(response_text)
            if "result" in json_response:
                result_json = json.dumps(json_response["result"], ensure_ascii=False).encode('utf-8')
                logger.info(f"[OCR API] 提取result字段成功，result大小: {len(result_json)} bytes")
                return result_json
            else:
                logger.error(f"[OCR API] OCR API响应缺少'result'字段，响应键: {list(json_response.keys())}")
                raise ValueError("OCR API response missing 'result' field")
        except requests.exceptions.RequestException as e:
            ocr_duration = time.time() - ocr_start
            logger.error(f"[OCR API] OCR API调用失败（网络错误），耗时: {ocr_duration:.2f}秒，错误: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            ocr_duration = time.time() - ocr_start
            logger.error(f"[OCR API] OCR API调用失败，耗时: {ocr_duration:.2f}秒，错误: {str(e)}", exc_info=True)
            raise
    
    def _convert_textin_position_to_ragflow(self, position: List[int], page_id: int) -> List[int]:
        """
        将Textin的8个角点坐标转换为RAGFlow的5个数字格式
        
        Textin格式: [左上x, 左上y, 右上x, 右上y, 右下x, 右下y, 左下x, 左下y]
        RAGFlow格式: [page_id, left, right, top, bottom] (与add_positions和Infinity对齐)
        
        Args:
            position: Textin的position数组（8个数字）
            page_id: 页码（从1开始）
        
        Returns:
            RAGFlow格式的position数组（5个数字）：[page_id, left, right, top, bottom]
        """
        if not position or len(position) < 8:
            logger.debug(f"[位置转换] 位置信息不完整，使用默认值: page_id={page_id}, position长度={len(position) if position else 0}")
            return [page_id, 0, 0, 0, 0]
        
        # 提取4个角点坐标
        top_left_x, top_left_y = position[0], position[1]
        top_right_x, top_right_y = position[2], position[3]
        bottom_right_x, bottom_right_y = position[4], position[5]
        bottom_left_x, bottom_left_y = position[6], position[7]
        
        # 计算边界框（与add_positions和Infinity格式对齐）
        left = min(top_left_x, bottom_left_x)  # 左边界
        right = max(top_right_x, bottom_right_x)  # 右边界
        top = min(top_left_y, top_right_y)  # 上边界
        bottom = max(bottom_left_y, bottom_right_y)  # 下边界
        
        ragflow_position = [page_id, int(left), int(right), int(top), int(bottom)]
        logger.debug(f"[位置转换] Textin位置 {position[:4]}... -> RAGFlow位置 {ragflow_position}")
        return ragflow_position
        
    def parse(self, filename: str, binary: bytes, from_page: int = 0, to_page: int = 100000, **kwargs) -> List[Dict[str, Any]]:
        """
        自定义解析逻辑 - 支持PDF和JSON文件
        """
        logger.info(f"[解析入口] Custom parser开始处理文件: {filename}, 文件大小: {len(binary)} bytes")
        
        # 检查文件类型
        file_ext = os.path.splitext(filename)[1].lower()
        logger.info(f"[解析入口] 文件类型: {file_ext}")
        
        if filename.lower().endswith('.pdf'):
            logger.info(f"[解析入口] 识别为PDF文件，调用PDF解析流程")
            return self._parse_pdf_file(filename, binary, **kwargs)
        elif filename.lower().endswith('.json'):
            logger.info(f"[解析入口] 识别为JSON文件，调用JSON解析流程")
            return self._parse_json_file(filename, binary, **kwargs)
        else:
            # 对于不支持的文件类型，返回空列表
            logger.warning(f"[解析入口] Custom parser仅支持PDF和JSON文件，当前文件: {filename} (扩展名: {file_ext})")
            return []
    
    def _parse_pdf_file(self, filename: str, binary: bytes, **kwargs) -> List[Dict[str, Any]]:
        """
        解析PDF文件 - 使用OCR API或缓存
        """
        try:
            logger.info(f"[PDF解析] 开始解析PDF文件: {filename}, 文件大小: {len(binary)} bytes")
            
            # 1. 计算PDF的MD5值
            md5 = self._calculate_pdf_md5(binary)
            logger.info(f"[PDF解析] PDF文件MD5: {md5}")
            
            # 2. 检查缓存
            logger.info(f"[PDF解析] 检查缓存 (bucket: {self.json_cache_bucket}, key: {md5}.json)")
            cached_json = self._get_cached_json(md5)
            if cached_json:
                logger.info(f"[PDF解析] ✓ 缓存命中，使用缓存的JSON: {filename} (MD5: {md5}), JSON大小: {len(cached_json)} bytes")
                chunks = self._parse_json_file(filename, cached_json, **kwargs)
                logger.info(f"[PDF解析] 缓存JSON解析完成，生成 {len(chunks)} 个chunks")
                return chunks
            
            logger.info(f"[PDF解析] ✗ 缓存未命中，需要调用OCR API")
            
            # 3. 调用OCR API
            if not self.ocr_client:
                logger.error(f"[PDF解析] OCR客户端未初始化，无法解析PDF: {filename}")
                return []
            
            logger.info(f"[PDF解析] 调用OCR API解析PDF: {filename} (MD5: {md5})")
            import time
            ocr_start_time = time.time()
            json_binary = self._call_ocr_api(binary)
            ocr_duration = time.time() - ocr_start_time
            logger.info(f"[PDF解析] OCR API调用完成，耗时: {ocr_duration:.2f}秒，返回JSON大小: {len(json_binary)} bytes")
            
            # 4. 保存到缓存
            logger.info(f"[PDF解析] 保存PDF和JSON到缓存 (PDF bucket: {self.pdf_cache_bucket}, JSON bucket: {self.json_cache_bucket})")
            self._save_to_cache(md5, binary, json_binary)
            logger.info(f"[PDF解析] ✓ 缓存保存完成")
            
            # 5. 解析JSON
            logger.info(f"[PDF解析] 开始解析OCR返回的JSON")
            chunks = self._parse_json_file(filename, json_binary, **kwargs)
            logger.info(f"[PDF解析] PDF解析完成: {filename}, 生成 {len(chunks)} 个chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"[PDF解析] PDF文件解析失败: {filename}, 错误: {str(e)}", exc_info=True)
            return []
    
    def _parse_json_file(self, filename: str, binary: bytes, **kwargs) -> List[Dict[str, Any]]:
        """
        解析JSON格式文件（基于Textin格式）
        """
        try:
            logger.info(f"[JSON解析] 开始解析JSON文件: {filename}, JSON大小: {len(binary)} bytes")
            
            # 解析JSON数据
            json_data = json.loads(binary.decode('utf-8'))
            logger.info(f"[JSON解析] JSON解析成功，根对象类型: {type(json_data).__name__}")
            
            # 重置章节状态
            self.title_hierarchy = []
            self.current_hierarchy = []
            logger.info(f"[JSON解析] 章节层级状态已重置")
            
            # 检查JSON格式
            if not isinstance(json_data, dict) or 'detail' not in json_data:
                logger.warning(f"[JSON解析] JSON文件 {filename} 格式不正确，缺少detail字段")
                return []
            
            if not isinstance(json_data['detail'], list):
                logger.warning(f"[JSON解析] JSON文件 {filename} 的detail字段不是列表类型，实际类型: {type(json_data['detail']).__name__}")
                return []
            
            detail_count = len(json_data['detail'])
            logger.info(f"[JSON解析] detail数组包含 {detail_count} 个数据项")
            
            # 处理所有数据项
            chunks = []
            processed_count = 0
            skipped_count = 0
            
            for idx, item in enumerate(json_data['detail']):
                if not isinstance(item, dict):
                    skipped_count += 1
                    logger.debug(f"[JSON解析] 跳过第 {idx+1} 项（非字典类型）")
                    continue
                
                chunk = self._process_json_item(item, filename)
                if chunk:
                    chunks.append(chunk)
                    processed_count += 1
                else:
                    skipped_count += 1
            
            logger.info(f"[JSON解析] JSON文件 {filename} 解析完成: 处理 {processed_count} 项，跳过 {skipped_count} 项，生成 {len(chunks)} 个chunks")
            return chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"[JSON解析] JSON文件 {filename} 解析失败（JSON格式错误）: {str(e)}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"[JSON解析] 处理JSON文件 {filename} 时发生错误: {str(e)}", exc_info=True)
            return []
    
    
    def _process_json_item(self, item: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """
        处理JSON数据项
        """
        # 支持两种字段名：type 和 sub_type
        item_type = item.get('type', '') or item.get('sub_type', '')
        page_id = item.get('page_id', 'N/A')
        sub_type = item.get('sub_type', 'N/A')
        
        logger.debug(f"[JSON项处理] 处理项: type={item_type}, sub_type={sub_type}, page_id={page_id}")
        
        if item_type in ['paragraph', 'text', 'text_title']:
            chunk = self._process_paragraph_item(item, filename)
            if chunk:
                logger.debug(f"[JSON项处理] ✓ 段落项处理成功: page_id={page_id}, sub_type={sub_type}")
            return chunk
        elif item_type == 'table':
            chunk = self._process_table_item(item, filename)
            if chunk:
                logger.debug(f"[JSON项处理] ✓ 表格项处理成功: page_id={page_id}")
            return chunk
        # elif item_type == 'image':
        #     return self._process_image_item(item, filename)
        else:
            # 其他类型暂时跳过
            logger.debug(f"[JSON项处理] ✗ 跳过未支持的类型: type={item_type}")
            return None
    
    def _create_base_doc(self, filename: str) -> Dict[str, Any]:
        """
        创建基础文档结构，包含RAGFlow标准字段
        """
        # 从文件名提取文档名（去除扩展名）
        doc_name = re.sub(r"\.[a-zA-Z]+$", "", filename)
        if not doc_name:
            doc_name = "Untitled Document"
        
        doc = {
            "docnm_kwd": doc_name,  # 使用文档名而不是完整文件名
            "title_tks": rag_tokenizer.tokenize(doc_name)
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        
        return doc
    
    def _process_paragraph_item(self, item: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """
        处理段落项
        """
        text = item.get('text', '').strip()
        if not text:
            return None
        
        sub_type = item.get('sub_type', '')
        outline_level = item.get('outline_level', -1)
        
        # 创建基础文档结构
        doc = self._create_base_doc(filename)
        
        # 处理标题
        if sub_type == 'text_title' and outline_level >= 0:
            return self._process_title_item(item, filename, doc)
        elif sub_type == 'text':
            # 处理普通文本
            return self._process_text_item(item, filename, doc)
    
    def _process_title_item(self, item: Dict[str, Any], filename: str, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理标题项 - 更新章节层级
        """
        title_text = item.get('text', '').strip()
        if not title_text:
            return None
        
        outline_level = int(item.get('outline_level', 0))
        
        # 更新层级结构
        if outline_level < len(self.current_hierarchy):
            self.current_hierarchy = self.current_hierarchy[:outline_level]
        
        if outline_level >= len(self.current_hierarchy):
            self.current_hierarchy.append(title_text)
        else:
            self.current_hierarchy[outline_level] = title_text
        
        # 更新标题层级
        self.title_hierarchy = self.current_hierarchy.copy()
        
        # 创建标题chunk
        chunk = doc.copy()
        
        # 添加位置信息（根据Textin格式转换）
        page_id = item.get('page_id', 1)  # Textin的page_id从1开始
        position = item.get('position', [])  # Textin的position是8个数字（4个角点）
        
        # 转换为RAGFlow格式
        if position and len(position) >= 8:
            ragflow_position = self._convert_textin_position_to_ragflow(position, page_id)
            chunk["position_int"] = [ragflow_position]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [ragflow_position[3]]  # top坐标（格式：[page_id, left, right, top, bottom]）
        else:
            # 默认位置信息（格式：[page_id, left, right, top, bottom]）
            chunk["position_int"] = [[page_id, 0, 0, 0, 0]]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [0]
        
        chunk.update({
            "doc_type_kwd": "title"  # 文档类型
        })
        
        # 使用RAGFlow标准分词
        tokenize(chunk, title_text, False)  # 假设是中文文档
        
        # 位置信息已在上面设置，不需要再调用add_positions
        
        # 为章节标题添加重要关键词字段，提高检索权重
        #if title_text:
        #    chunk["important_kwd"], chunk["important_tks"] = self._generate_keywords(
        #        title_text, topn=3, context=f"title '{title_text}'"
        #    )
        
        return chunk
    
    def _process_text_item(self, item: Dict[str, Any], filename: str, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理普通文本项
        """
        # 过滤掉包含"参考文献"的title_hierarchy
        if self.title_hierarchy and any("参考文献" in title for title in self.title_hierarchy):
            return None
            
        text = item.get('text', '').strip()
        if not text:
            return None
        
        # 只记录section_title，不预先拼接内容，将文件名拼入section
        doc_name = re.sub(r"\.[a-zA-Z]+$", "", filename)  # 去除文件扩展名
        if self.title_hierarchy:
            section_title = f"{doc_name} > {' > '.join(self.title_hierarchy)}"
        else:
            section_title = doc_name

        # 创建chunk，基于基础文档结构
        chunk = doc.copy()
        
        # 添加位置信息（根据Textin格式转换）
        page_id = item.get('page_id', 1)  # Textin的page_id从1开始
        position = item.get('position', [])  # Textin的position是8个数字（4个角点）
        
        # 转换为RAGFlow格式
        if position and len(position) >= 8:
            ragflow_position = self._convert_textin_position_to_ragflow(position, page_id)
            chunk["position_int"] = [ragflow_position]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [ragflow_position[3]]  # top坐标（格式：[page_id, left, right, top, bottom]）
        else:
            # 默认位置信息（格式：[page_id, left, right, top, bottom]）
            chunk["position_int"] = [[page_id, 0, 0, 0, 0]]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [0]
        
        # 构建包含章节标题的完整内容
        if section_title:
            content_with_weight = f"[{section_title}]\n{text}"
        else:
            content_with_weight = text
        
        chunk.update({
            "content_with_weight": content_with_weight,  # 包含章节标题的完整内容
            "section_title": section_title,  # 记录章节标题
            "doc_type_kwd": "text"  # 文档类型
        })
        
        # 使用RAGFlow标准分词 - 使用原始文本
        tokenize(chunk, text, False)  # 假设是中文文档
        
        # 位置信息已在上面设置，不需要再调用add_positions
        
        # 为包含章节标题的文本内容添加重要关键词字段
        if section_title:
            # 构建用于关键词生成的内容（章节标题）
            #content_for_keywords = f"{section_title}\n{text}"
            content_for_keywords = f"{section_title}"
            chunk["important_kwd"], chunk["important_tks"] = self._generate_keywords(
                content_for_keywords, topn=5, context=f"text_content '{section_title}'"
            )
        
        return chunk
    
    def _process_table_item(self, item: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """
        处理表格项
        """
        cells = item.get('cells', [])
        if not isinstance(cells, list) or not cells:
            return None
        
        # 构建表格内容
        table_content = []
        for cell in cells:
            if isinstance(cell, dict) and 'text' in cell:
                table_content.append(str(cell.get('text', '')))
        
        if not table_content:
            return None
        
        # 构建表格文本
        table_text = " | ".join(table_content)
        
        # 构建增强内容
        section_title = " > ".join(self.title_hierarchy) if self.title_hierarchy else ""
        if section_title:
            enhanced_content = f"[{section_title}]\n[表格] {table_text}"
        else:
            enhanced_content = f"[表格] {table_text}"
        
        # 创建基础文档结构
        doc = self._create_base_doc(filename)
        
        # 创建chunk，基于基础文档结构
        chunk = doc.copy()
        
        # 添加位置信息（根据Textin格式转换）
        page_id = item.get('page_id', 1)  # Textin的page_id从1开始
        position = item.get('position', [])  # Textin的position是8个数字（4个角点）
        
        # 转换为RAGFlow格式
        if position and len(position) >= 8:
            ragflow_position = self._convert_textin_position_to_ragflow(position, page_id)
            chunk["position_int"] = [ragflow_position]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [ragflow_position[3]]  # top坐标（格式：[page_id, left, right, top, bottom]）
        else:
            # 默认位置信息（格式：[page_id, left, right, top, bottom]）
            chunk["position_int"] = [[page_id, 0, 0, 0, 0]]
            chunk["page_num_int"] = [page_id]
            chunk["top_int"] = [0]
        
        chunk.update({
            "doc_type_kwd": "table"  # 文档类型
        })
        
        # 使用RAGFlow标准分词
        tokenize(chunk, enhanced_content, False)  # 假设是中文文档
        
        # 位置信息已在上面设置，不需要再调用add_positions
        
        return chunk
    
    def _apply_paper_merge_strategy(self, chunks: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """
        简单的合并策略 - 基于section_title分组合并
        理论上相同section_title的chunks应该合并在一起
        """
        if not chunks:
            logger.info(f"[合并策略] 没有chunks需要合并: {filename}")
            return chunks
        
        logger.info(f"[合并策略] 开始应用合并策略: {filename}, 原始chunks数: {len(chunks)}")
        
        # 按section_title分组
        grouped_chunks = {}
        title_chunks = []
        table_chunks = []  # 表格chunks单独处理，不参与合并
        
        for chunk in chunks:
            if chunk.get('doc_type_kwd') == 'title':
                title_chunks.append(chunk)
            elif chunk.get('doc_type_kwd') == 'table':
                # 表格chunks单独处理，不参与合并
                table_chunks.append(chunk)
            else:
                section_title = chunk.get('section_title', '')
                if section_title not in grouped_chunks:
                    grouped_chunks[section_title] = []
                grouped_chunks[section_title].append(chunk)
        
        logger.info(f"[合并策略] 分组统计: 标题chunks={len(title_chunks)}, 表格chunks={len(table_chunks)}, 文本分组数={len(grouped_chunks)}")
        
        # 合并每个分组的chunks
        merged_chunks = []
        
        # 先添加标题chunks
        merged_chunks.extend(title_chunks)
        logger.debug(f"[合并策略] 添加了 {len(title_chunks)} 个标题chunks")
        
        # 添加表格chunks（每个表格单独一个chunk，不合并）
        merged_chunks.extend(table_chunks)
        logger.debug(f"[合并策略] 添加了 {len(table_chunks)} 个表格chunks（不合并）")
        
        # 合并文本chunks
        merged_count = 0
        single_count = 0
        for section_title, text_chunks in grouped_chunks.items():
            if not text_chunks:
                continue
                
            if len(text_chunks) == 1:
                # 只有一个chunk，也需要确保content_with_weight包含标题
                single_chunk = copy.deepcopy(text_chunks[0])
                if section_title and not single_chunk.get('content_with_weight', '').startswith('['):
                    # 如果content_with_weight还没有包含标题，则添加
                    single_chunk['content_with_weight'] = f"[{section_title}]\n{single_chunk.get('content_with_weight', '')}"
                merged_chunks.append(single_chunk)
                single_count += 1
            else:
                # 多个chunks，需要合并
                logger.debug(f"[合并策略] 合并分组 '{section_title}': {len(text_chunks)} 个chunks")
                merged_chunk = self._create_merged_chunk(text_chunks, section_title)
                if merged_chunk:
                    merged_chunks.append(merged_chunk)
                    merged_count += 1
        
        logger.info(f"[合并策略] 合并完成: {filename}, {len(chunks)} -> {len(merged_chunks)} chunks (合并了 {merged_count} 个分组，保留 {single_count} 个单chunk分组)")
        return merged_chunks
    
    def _create_merged_chunk(self, text_chunks: List[Dict[str, Any]], section_title: str) -> Optional[Dict[str, Any]]:
        """
        合并相同section_title的文本chunks
        """
        if not text_chunks:
            return None
        
        # 使用第一个chunk作为基础
        base_chunk = copy.deepcopy(text_chunks[0])
        
        # 收集所有文本内容
        text_contents = []
        for chunk in text_chunks:
            # 从content_with_weight中提取原始文本内容
            content_with_weight = chunk.get('content_with_weight', '')
            # 如果包含章节标题格式，提取原始文本
            if content_with_weight.startswith('[') and ']\n' in content_with_weight:
                content = content_with_weight.split(']\n', 1)[1] if ']\n' in content_with_weight else content_with_weight
            else:
                content = content_with_weight
            if content.strip():
                text_contents.append(content.strip())
        
        if not text_contents:
            return None
        
        # 合并文本内容
        merged_text = '\n'.join(text_contents)
        
        # 构建最终的content_with_weight
        if section_title:
            final_content = f"[{section_title}]\n{merged_text}"
        else:
            final_content = merged_text
        
        # 更新chunk内容
        base_chunk['content_with_weight'] = final_content  # 包含标题的完整内容
        
        # 重新分词处理
        try:
            # 清除旧的分词结果
            for key in ['content_ltks', 'content_sm_ltks']:
                if key in base_chunk:
                    del base_chunk[key]
            
            # 重新分词 - 使用最终内容
            tokenize(base_chunk, final_content, False)
            
        except Exception as e:
            logger.warning(f"Failed to retokenize merged chunk: {e}")
        
        return base_chunk
    
    def _merge_content_without_duplicate_titles(self, content_list: List[str]) -> str:
        """
        合并内容，避免重复标题
        每个内容项都包含章节标题，合并时只保留第一个标题，后续内容去掉重复的标题部分
        """
        if not content_list:
            return ""
        
        if len(content_list) == 1:
            return content_list[0]
        
        # 提取第一个内容的标题部分
        first_content = content_list[0]
        section_title = ""
        
        # 查找章节标题（以 " > " 分隔的层级结构）
        lines = first_content.split('\n')
        if lines and ' > ' in lines[0]:
            section_title = lines[0]
        
        # 合并所有内容
        merged_parts = []
        
        for i, content in enumerate(content_list):
            if i == 0:
                # 第一个内容保留完整
                merged_parts.append(content)
            else:
                # 后续内容去掉重复的标题部分
                cleaned_content = self._extract_clean_text_content(content)
                if cleaned_content.strip():
                    merged_parts.append(cleaned_content)
        
        return '\n'.join(merged_parts)
    
    def _extract_clean_text_content(self, content: str) -> str:
        """
        从包含章节标题的内容中提取纯文本内容
        标题用方括号[]包围，去掉方括号内的标题部分
        """
        lines = content.split('\n')
        if not lines:
            return content
        
        cleaned_lines = []
        for line in lines:
            # 如果行以[开头且以]结尾，认为是标题，跳过
            if line.strip().startswith('[') and line.strip().endswith(']'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    # def _process_image_item(self, item: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
    #     """
    #     处理图像项 - 暂时注释掉
    #     """
    #     text = item.get('text', '').strip()
    #     if not text:
    #         return None
    #     
    #     # 构建增强内容
    #     section_title = " > ".join(self.title_hierarchy) if self.title_hierarchy else ""
    #     enhanced_content = f"[图像] {text}"
    #     if section_title:
    #         enhanced_content = f"[章节: {section_title}] {enhanced_content}"
    #     
    #     # 创建基础文档结构
    #     doc = self._create_base_doc(filename)
    #     
    #     # 创建chunk，基于基础文档结构
    #     chunk = doc.copy()
    #     
    #     # 添加位置信息
    #     position = item.get('position', [])
    #     if position and len(position) >= 4:
    #         chunk["position_int"] = [position]
    #         chunk["page_num_int"] = [item.get('page_id', 1)]
    #     else:
    #         chunk["position_int"] = [[item.get('page_id', 1), 0, 0, 0, 0]]
    #         chunk["page_num_int"] = [item.get('page_id', 1)]
    #     
    #     chunk.update({
    #         "content": enhanced_content,
    #         "page_number": item.get('page_id', 1),
    #         "paragraph_id": item.get('paragraph_id', 0),
    #         "sub_type": "image",
    #         "section_title": section_title,
    #         "position": position,
    #         "content_type": "image",
    #         "doc_type_kwd": "image",  # 文档类型
    #         "image_url": item.get('image_url', '')
    #     })
    #     
    #     # 使用RAGFlow标准分词
    #     tokenize(chunk, enhanced_content, False)  # 假设是中文文档
    #     
    #     return chunk
    


def chunk(filename: str, binary: Optional[bytes] = None, from_page: int = 0, to_page: int = 100000,
          lang: str = "Chinese", callback=None, **kwargs) -> List[Dict[str, Any]]:
    """
    自定义解析器的主入口函数
    支持JSON格式文件和章节标题提取
    """
    logger.info(f"Custom parser chunk function called for: {filename}")
    
    try:
        # 获取解析配置
        parser_config = kwargs.get("parser_config", {})
        custom_config = parser_config.get("custom_config", {})
        
        # 添加tenant_id到custom_config中，用于LLM模型初始化
        if "tenant_id" in kwargs:
            custom_config["tenant_id"] = kwargs["tenant_id"]
        
        # 确保bucket配置使用连字符（如果custom_config中没有指定，使用默认值）
        if "pdf_cache_bucket" not in custom_config:
            custom_config["pdf_cache_bucket"] = "pdf-cache"
        if "json_cache_bucket" not in custom_config:
            custom_config["json_cache_bucket"] = "json-cache"
        
        logger.info(f"[解析器初始化] Parser config: {parser_config}")
        logger.info(f"[解析器初始化] Custom config bucket设置: pdf_cache_bucket={custom_config.get('pdf_cache_bucket', 'pdf-cache')}, json_cache_bucket={custom_config.get('json_cache_bucket', 'json-cache')}")
        
        # 创建自定义解析器实例
        parser = CustomPdfParser(custom_config=custom_config)
        
        # 执行解析
        chunks = parser.parse(filename, binary, from_page, to_page, **kwargs)
        
        logger.info(f"Raw chunks generated: {len(chunks)}")
        

        # 后处理：添加元数据等
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append(chunk)
        processed_chunks = parser._apply_paper_merge_strategy(chunks, filename)

        logger.info(f"Custom parser processed {len(processed_chunks)} chunks for {filename}")
        
        # 输出解析统计信息
        if processed_chunks:
            title_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") == "title"]
            text_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") == "text"]
            table_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") == "table"]
            image_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") == "image"]
            
            logger.info(f"解析统计 - 标题: {len(title_chunks)}, 文本: {len(text_chunks)}, 表格: {len(table_chunks)}, 图像: {len(image_chunks)}")
            print(f"解析统计 - 标题: {len(title_chunks)}, 文本: {len(text_chunks)}, 表格: {len(table_chunks)}, 图像: {len(image_chunks)}")
        
        # 删除section_title字段，避免Infinity数据库插入错误
        for chunk in processed_chunks:
            if "section_title" in chunk:
                del chunk["section_title"]
        
        # 过滤掉标题类型的chunks，只返回文本内容
        filtered_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") != "title"]
        return filtered_chunks
        
    except Exception as e:
        logger.error(f"Custom parser error for {filename}: {str(e)}")
        if callback:
            callback(-1, f"Custom parser error: {str(e)}")
        raise
