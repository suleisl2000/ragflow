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
import base64
import tempfile
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path

from rag.nlp import rag_tokenizer, tokenize, add_positions, tokenize_chunks, bullets_category, title_frequency
from api.db import ParserType, LLMType
from api.db.services.llm_service import LLMBundle         
from rag.prompts.generator import keyword_extraction
from rag.utils.storage_factory import STORAGE_IMPL
from api.utils.configs import read_config

logger = logging.getLogger(__name__)

# 标题模式列表（从 gen_title_report.py 复制）
TITLE_PATTERNS = [
    # 模式0: 以1个或多个空格开头的markdown标题，或作为兜底匹配所有不匹配其他模式的标题（优先级0，与模式1同级）
    (
        r"^.+",
        0,
        None
    ),
    # 模式1: 第X章/节/条等（优先级0）
    (
        r"^第[零一二三四五六七八九十百千0-9]+(分?编|部分|篇|章|节|条)",
        0,
        None
    ),
    # 模式2: 一、二、三、（中文数字+顿号/全角点号/空格，优先级1）
    # 支持：一、 一． 一 （中文数字后跟顿号、全角点号或空格）
    (
        r"^([零一二三四五六七八九十百千]+|[一二三四五六七八九十]+)([、．]|\s+)",
        1,
        None
    ),
    # 模式3: （一）（二）（中文括号+中文数字，优先级2）
    (
        r"^[（(]([零一二三四五六七八九十百]+|[一二三四五六七八九十]+)[）)]",
        2,
        None
    ),
    # 模式4: "数字+空格"、"数字+、"、"数字+."（优先级3）
    # 注意：不能匹配数字x.x.x格式（由模式5-9处理）
    # 注意：不能匹配"数字+空格+右括号"格式（由模式9处理）
    # 匹配格式：数字+空格/点号/顿号+标题内容
    # 负向前瞻：排除"数字+空格+1-3位数字"（短数字，如"2 10个"），但允许"数字+空格+4位数字"（年份，如"2 2019"）
    # 支持的格式示例："3 局限性和未来的方向"、"2 2019 年更新共识的主要内容"、"1. 适应证"、"1、 病史"
    (
        r"^([0-9]{1,2})(\s+|[、.．]\s*)(?![0-9]{1,3}(?![0-9])|[）)]|$)",
        3,
        None
    ),
    # 模式5: 数字x.x格式（优先级4）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        4,
        1
    ),
    # 模式6: 数字x.x.x格式（优先级5）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        5,
        2
    ),
    # 模式7: 数字x.x.x.x格式（优先级6）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        6,
        3
    ),
    # 模式8: 数字x.x.x.x.x格式（优先级7）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        7,
        4
    ),
    # 模式9: 数字（1）格式或数字+空格+右括号格式（优先级8）
    # 支持: "（1）"、"(1)"、"1 ）"、"2 ）" 等格式
    (
        r"^(?:[（(]([0-9]{1,2})[）)]|([0-9]{1,2})\s+[）)])",
        8,
        None
    ),
    # 模式10: "问题1"、"问题一"、"临床问题1"、"临床问题 1"、"问题 1"、"陈述1"、"陈述 1"格式（优先级9）
    # 支持: "问题1"、"问题一"、"问题 1"、"临床问题1"、"临床问题 1"、"临床问题一"、"陈述1"、"陈述 1"、"陈述一" 等格式
    (
        r"^((?:临床)?(?:问题|陈述))\s*([0-9]+|[零一二三四五六七八九十百千]+)",
        9,
        None
    ),
    # 模式11: "推荐意见1"、"推荐意见一"格式（优先级10，低于模式10）
    # 支持: "推荐意见1"、"推荐意见一"、"推荐意见2"、"推荐意见二" 等格式
    # 注意：优先级低于模式10（问题1），由于正则表达式互斥（模式10以"问题"开头，模式11以"推荐意见"开头），不会同时匹配
    (
        r"^推荐意见([0-9]+|[零一二三四五六七八九十百千]+)",
        10,
        None
    ),
]

# 常见的前言性标题（一级标题）
# 包含简体字和繁体字版本
PREFACE_TITLES = {
    '目次', '目录', '前言', '引言', '概述', '摘要', '参考文献', '参考文献：',
    '參考文献', '參考文献：',  # 繁体字版本
    '附录', '致谢', '后记', '编写说明', '展望',
    '附錄', '致謝', '後記', '編寫說明', '展望',  # 繁体字版本
    '【摘要】', '【Abstract】'  # 带方括号的格式
}


def not_bullet(line: str) -> bool:
    """判断是否不是有效的标题编号"""
    # 只检查最基本的无效格式，严格按照模式匹配来判断
    patt = [
        r"^0$",  # 单独的0
        r"^[0-9]+\.{2,}",  # 数字+多个点
    ]
    return any([re.match(p, line.strip()) for p in patt])


def get_title_pattern_info(title_text: str) -> tuple:
    """
    获取标题的模式信息
    
    Args:
        title_text: 标题文本
    
    Returns:
        (priority, dot_count) 元组，如果不匹配任何模式返回 (-1, None)
        priority: 模式优先级（数字越小优先级越高）
        dot_count: 点号数量（仅用于数字x.x格式，其他模式为None）
    """
    # 保留前导空格（用于匹配模式0），只去掉尾随空格
    content_with_leading_space = title_text.rstrip()
    # 去掉所有前后空格（用于匹配其他模式）
    content = title_text.strip()
    
    # 如果是前言性标题，返回最高优先级（0）
    # 支持中间有空格的情况（去掉所有空格后比较）
    content_no_spaces = content.replace(' ', '').replace('　', '')  # 去掉普通空格和全角空格
    content_no_spaces_no_colon = content_no_spaces.rstrip('：:')
    if (content in PREFACE_TITLES or 
        content.rstrip('：:') in PREFACE_TITLES or
        content_no_spaces in PREFACE_TITLES or
        content_no_spaces_no_colon in PREFACE_TITLES):
        return (0, None)
    
    # 先检查其他模式（不需要前导空格）
    for pattern, priority, dot_count in TITLE_PATTERNS[1:]:
        match = re.match(pattern, content)
        if match and not not_bullet(content):
            return (priority, dot_count)
    
    # 如果没有匹配任何其他模式，检查是否匹配模式0
    # 模式0匹配：1) 有前导空格的标题，或 2) 所有不匹配其他模式的标题（作为兜底）
    pattern0, priority0, dot_count0 = TITLE_PATTERNS[0]
    # 优先检查有前导空格的情况
    if content_with_leading_space != content:  # 有前导空格
        match0 = re.match(pattern0, content_with_leading_space)
        if match0 and not not_bullet(content_with_leading_space):
            return (priority0, dot_count0)
    # 如果没有前导空格，但也不匹配其他模式，也返回模式0作为兜底
    if content:
        match0 = re.match(pattern0, content)
        if match0 and not not_bullet(content):
            return (priority0, dot_count0)
    
    # 如果没有匹配任何模式，返回-1
    return (-1, None)


def get_title_pattern_priority(title_text: str) -> int:
    """获取标题的模式优先级（兼容旧接口）"""
    priority, _ = get_title_pattern_info(title_text)
    return priority


def is_valid_title(title_text: str) -> bool:
    """判断标题是否匹配任何标题模式"""
    return get_title_pattern_priority(title_text) >= 0


def adjust_title_levels(titles: list) -> list:
    """
    调整标题层级：完全按照标题模式优先级确定绝对Level，不考虑markdown层级
    
    Args:
        titles: 标题列表，每个元素包含 'content', 'level', 'line', 'file'
    
    Returns:
        调整后的标题列表
    """
    if not titles:
        return titles
    
    adjusted_titles = titles.copy()
    
    # 保存原始 markdown 层级
    for title in adjusted_titles:
        title['original_md_level'] = title['level']
    
    # 收集所有标题的模式信息
    title_infos = []
    for idx, title in enumerate(adjusted_titles):
        priority, dot_count = get_title_pattern_info(title['content'])
        title_infos.append((idx, title, priority, dot_count))
    
    # 过滤掉没有匹配任何模式的标题（priority < 0）
    valid_titles = [(idx, title, priority, dot_count) for idx, title, priority, dot_count in title_infos if priority >= 0]
    
    if not valid_titles:
        # 所有标题都没有匹配模式，保持原层级
        return adjusted_titles
    
    # 对于模式0，只保留紧挨在模式2（"一、"）之前的匹配标题
    # 如果有多个"一、"标题，每个"一、"之前如果有匹配模式0的标题都要保留
    # 但是"二、"或其他模式2之前的模式0不保留
    pattern1_regex = TITLE_PATTERNS[1][0]  # 模式1的正则表达式
    pattern2_regex = TITLE_PATTERNS[2][0]  # 模式2的正则表达式（"一、"、"二、"等）
    
    # 找到所有模式2（"一、"）的标题位置
    keep_pattern0_indices = set()
    for i, (idx, title, priority, dot_count) in enumerate(valid_titles):
        if priority == 1:  # 模式2的优先级是1
            # 检查是否是"一、"（匹配模式2的正则，且内容以"一、"开头）
            title_content = title['content'].strip()
            if re.match(pattern2_regex, title_content):
                # 进一步检查是否以"一、"、"一．"或"一 "开头（而不是"二、"、"三、"等）
                if (title_content.startswith('一、') or title_content.startswith('一．') or 
                    title_content.startswith('一 ')):
                    # 检查它前面的标题是否是模式0
                    if i > 0:
                        prev_idx, prev_title, prev_priority, prev_dot_count = valid_titles[i - 1]
                        # 如果前一个标题是模式0（优先级0且不匹配模式1）
                        if prev_priority == 0 and not re.match(pattern1_regex, prev_title['content'].strip()):
                            # 标记为保留
                            keep_pattern0_indices.add(prev_idx)
    
    # 过滤掉所有未标记保留的模式0标题
    # 注意：前言性标题（匹配PREFACE_TITLES的标题）应该保留，不应该被过滤
    filtered_titles = []
    for idx, title, priority, dot_count in valid_titles:
        # 如果是模式0（优先级0且不匹配模式1）
        if priority == 0 and not re.match(pattern1_regex, title['content'].strip()):
            # 检查是否是前言性标题
            content = title['content'].strip()
            content_no_spaces = content.replace(' ', '').replace('　', '')
            content_no_spaces_no_colon = content_no_spaces.rstrip('：:')
            is_preface = (content in PREFACE_TITLES or 
                         content.rstrip('：:') in PREFACE_TITLES or
                         content_no_spaces in PREFACE_TITLES or
                         content_no_spaces_no_colon in PREFACE_TITLES)
            
            # 如果是前言性标题，保留；否则只有当它被标记为保留时（即紧挨在"一、"之前），才保留
            if not is_preface and idx not in keep_pattern0_indices:
                continue
        filtered_titles.append((idx, title, priority, dot_count))
    valid_titles = filtered_titles
    
    # 检查模式0和模式1是否在同一Level（互斥检查）
    # 注意：custom_parser.py不返回has_conflict，但保留检查逻辑以便将来使用
    has_pattern0 = False
    has_pattern1 = False
    
    for idx, title, priority, dot_count in valid_titles:
        if priority == 0:
            if re.match(pattern1_regex, title['content'].strip()):
                has_pattern1 = True
            else:
                has_pattern0 = True
    
    # 如果同时存在模式0和模式1，则冲突（这里只做检查，不返回）
    # has_conflict = has_pattern0 and has_pattern1
    
    # 创建 (priority, dot_count) 对，用于排序和映射
    priority_dot_pairs = set()
    for idx, title, priority, dot_count in valid_titles:
        priority_dot_pairs.add((priority, dot_count))
    
    # 排序：优先级越小越好，对于数字x.x格式，点号数量越少越好
    sorted_pairs = sorted(priority_dot_pairs, key=lambda x: (x[0], x[1] if x[1] is not None else 0))
    
    # 创建 (priority, dot_count) 到层级的映射
    # Level从2开始（对应markdown的##）
    pair_to_level = {}
    for i, (priority, dot_count) in enumerate(sorted_pairs):
        pair_to_level[(priority, dot_count)] = 2 + i
    
    # 应用层级映射
    for idx, title, priority, dot_count in valid_titles:
        key = (priority, dot_count)
        if key in pair_to_level:
            title['level'] = pair_to_level[key]
    
    # 只返回过滤后的标题（从valid_titles中提取）
    filtered_adjusted_titles = [title for idx, title, priority, dot_count in valid_titles]
    
    return filtered_adjusted_titles


def extract_titles_from_markdown(markdown_content: str) -> list:
    """
    从 markdown 内容中提取标题
    使用正则 r"^#{1,6}\\s+.*$" 匹配标题，层级由 # 的数量决定
    """
    titles = []
    lines = markdown_content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # 使用与 markdown_parser.py 相同的正则表达式
        if re.match(r"^#{1,6}\s+.*$", line):
            # 提取 # 的数量作为层级
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))  # # 的数量就是层级
                title_text = match.group(2).strip()
                
                # 过滤掉 HTML 注释
                if title_text.startswith('<!--') or title_text.endswith('-->'):
                    continue
                
                # 过滤掉空标题
                if not title_text:
                    continue
                
                # 过滤掉没有匹配任何标题模式的标题（除非是前言性标题）
                if not is_valid_title(title_text):
                    continue
                
                titles.append({
                    'content': title_text,
                    'level': level,
                    'line': line_num,
                })
    
    return titles


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


class PaddleOCRClient:
    """
    PaddleOCR HTTP API 客户端
    """
    def __init__(self, api_url: str = None):
        self.api_url = api_url or "http://localhost:8080/layout-parsing"
    
    def recognize(self, file_content: bytes, options: dict = None) -> str:
        """
        调用 PaddleOCR API 解析 PDF 文件
        
        Args:
            file_content: PDF 文件的二进制内容
            options: OCR 选项（可选，如 visualize=False）
        
        Returns:
            PaddleOCR API 返回的 markdown 文本（多页拼接）
        """
        if options is None:
            options = {}
        
        # 将文件内容转换为 Base64
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # 构建请求体
        payload = {
            "file": file_base64,
            "fileType": 0,  # 0=PDF, 1=图片
            "visualize": options.get("visualize", False)
        }
        
        # 发送请求
        response = requests.post(
            self.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=3600  # 60分钟超时
        )
        
        # 检查响应状态
        response.raise_for_status()
        result_data = response.json()
        
        if result_data.get("errorCode") != 0:
            error_msg = result_data.get("errorMsg", "Unknown error")
            raise ValueError(f"PaddleOCR API error: {error_msg}")
        
        # 提取所有页面的 markdown 文本并拼接
        layout_results = result_data.get("result", {}).get("layoutParsingResults", [])
        markdown_parts = []
        
        for res in layout_results:
            markdown_data = res.get("markdown", {})
            markdown_text = markdown_data.get("text", "")
            if markdown_text:
                # 修正图片路径（如果需要）
                markdown_text = markdown_text.replace('src="imgs/', 'src="images/imgs/')
                markdown_parts.append(markdown_text)
        
        # 拼接多页 markdown
        merged_markdown = "\n\n".join(markdown_parts)
        return merged_markdown


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
        # 根据 OCR 类型自动选择结果缓存 bucket
        if self.ocr_type == "paddleocr":
            self.result_cache_bucket = "paddleocr-cache"
        elif self.ocr_type == "textin":
            self.result_cache_bucket = "textin-cache"
        else:
            # 默认使用 textin-cache（向后兼容）
            self.result_cache_bucket = "textin-cache"
        logger.info(f"[解析器初始化] 缓存bucket配置: PDF={self.pdf_cache_bucket}, Result={self.result_cache_bucket} (OCR类型: {self.ocr_type})")
    
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
        """初始化OCR配置，默认使用 PaddleOCR，支持 Textin 作为备选，优先级：环境变量 > 配置文件 > custom_config"""
        try:
            # 读取配置文件中的 OCR 配置
            ocr_config = {}
            try:
                config = read_config('service_conf.yaml')
                ocr_config = config.get('ocr', {})
            except Exception as e:
                logger.debug(f"Failed to read OCR config from service_conf.yaml: {e}")
            
            # 检查是否有明确的 PaddleOCR 配置（优先级：环境变量 > 配置文件 > custom_config）
            paddle_api_url = (
                os.getenv('PADDLE_OCR_API_URL') or
                ocr_config.get('paddleocr', {}).get('api_url') or
                self.custom_config.get('paddle_ocr_api_url')
            )
            
            if paddle_api_url:
                # 如果明确配置了 PaddleOCR，尝试初始化
                try:
                    self.ocr_client = PaddleOCRClient(paddle_api_url)
                    self.ocr_type = "paddleocr"
                    logger.info(f"PaddleOCR client initialized successfully with API URL: {paddle_api_url}")
                except Exception as e:
                    # 如果明确配置了 PaddleOCR 但初始化失败，报错并禁用 OCR
                    logger.error(f"Failed to initialize PaddleOCR with configured URL {paddle_api_url}: {e}")
                    logger.error("OCR is disabled due to PaddleOCR initialization failure")
                    self.ocr_client = None
                    self.ocr_type = None
            else:
                # 如果没有明确配置 PaddleOCR，使用默认 URL
                default_paddle_url = "http://localhost:8080/layout-parsing"
                try:
                    self.ocr_client = PaddleOCRClient(default_paddle_url)
                    self.ocr_type = "paddleocr"
                    logger.info(f"PaddleOCR client initialized with default API URL: {default_paddle_url}")
                except Exception as e:
                    # 如果默认 PaddleOCR 初始化失败，尝试 Textin 作为备选
                    logger.warning(f"Default PaddleOCR initialization failed, trying Textin as fallback: {e}")
                    
                    # Textin OCR 配置（优先级：环境变量 > 配置文件 > custom_config）
                    app_id = (
                        os.getenv('TEXTIN_OCR_APP_ID') or
                        ocr_config.get('textin', {}).get('app_id') or
                        ocr_config.get('app_id') or  # 向后兼容：支持直接在 ocr 下的配置
                        self.custom_config.get('ocr_app_id')
                    )
                    secret_code = (
                        os.getenv('TEXTIN_OCR_SECRET_CODE') or
                        ocr_config.get('textin', {}).get('secret_code') or
                        ocr_config.get('secret_code') or  # 向后兼容：支持直接在 ocr 下的配置
                        self.custom_config.get('ocr_secret_code')
                    )
                    
                    # 获取 Textin API URL（优先级：配置文件 > custom_config > 默认值）
                    textin_api_url = (
                        ocr_config.get('textin', {}).get('api_url') or
                        self.custom_config.get('ocr_api_url') or
                        "https://api.textin.com/ai/service/v1/pdf_to_markdown"
                    )
                    
                    # 如果还没有，从custom_config读取（向后兼容）
                    if not app_id:
                        app_id = self.custom_config.get('ocr_app_id')
                    if not secret_code:
                        secret_code = self.custom_config.get('ocr_secret_code')
                    
                    # 如果配置了 Textin，创建 Textin OCR 客户端
                    if app_id and secret_code:
                        self.ocr_client = OCRClient(app_id, secret_code, textin_api_url)
                        self.ocr_type = "textin"
                        logger.info(f"Textin OCR client initialized successfully as fallback (API URL: {textin_api_url})")
                    else:
                        logger.warning("Neither PaddleOCR (default) nor Textin configured, PDF parsing will be disabled")
                        self.ocr_client = None
                        self.ocr_type = None
        except Exception as e:
            logger.warning(f"Failed to initialize OCR config: {e}")
            self.ocr_client = None
            self.ocr_type = None
    
    def _calculate_pdf_md5(self, binary: bytes) -> str:
        """计算PDF的MD5值"""
        return hashlib.md5(binary).hexdigest()
    
    def _get_cached_result(self, md5: str) -> Optional[bytes]:
        """从minio获取缓存的OCR结果（根据OCR类型自动选择bucket和扩展名）"""
        try:
            # 根据 OCR 类型确定扩展名和 bucket
            if self.ocr_type == "paddleocr":
                cache_key = f"{md5}.md"
                bucket = self.result_cache_bucket
            else:
                cache_key = f"{md5}.json"
                bucket = self.result_cache_bucket
            
            logger.debug(f"[缓存] 检查OCR结果缓存: bucket={bucket}, key={cache_key}, OCR类型={self.ocr_type}")
            
            if STORAGE_IMPL.obj_exist(bucket, cache_key):
                result_binary = STORAGE_IMPL.get(bucket, cache_key)
                logger.info(f"[缓存] ✓ OCR结果缓存命中: MD5={md5}, OCR类型={self.ocr_type}, 大小={len(result_binary)} bytes")
                return result_binary
            else:
                logger.debug(f"[缓存] ✗ OCR结果缓存未命中: MD5={md5}, OCR类型={self.ocr_type}")
        except Exception as e:
            logger.warning(f"[缓存] 获取OCR结果缓存失败: MD5={md5}, OCR类型={self.ocr_type}, 错误: {e}", exc_info=True)
        return None
    
    def _save_to_cache(self, md5: str, pdf_binary: bytes, result_binary: bytes):
        """保存PDF和OCR结果到minio（根据OCR类型自动选择bucket和扩展名）"""
        try:
            # 保存PDF到pdf-cache bucket
            pdf_key = f"{md5}.pdf"
            logger.debug(f"[缓存] 保存PDF到缓存: bucket={self.pdf_cache_bucket}, key={pdf_key}, 大小={len(pdf_binary)} bytes")
            STORAGE_IMPL.put(self.pdf_cache_bucket, pdf_key, pdf_binary)
            logger.info(f"[缓存] ✓ PDF保存成功: {self.pdf_cache_bucket}/{pdf_key}")
            
            # 根据 OCR 类型确定扩展名和 bucket
            if self.ocr_type == "paddleocr":
                result_key = f"{md5}.md"
                result_type = "Markdown"
            else:
                result_key = f"{md5}.json"
                result_type = "JSON"
            
            # 保存OCR结果到对应的bucket
            logger.debug(f"[缓存] 保存{result_type}到缓存: bucket={self.result_cache_bucket}, key={result_key}, 大小={len(result_binary)} bytes, OCR类型={self.ocr_type}")
            STORAGE_IMPL.put(self.result_cache_bucket, result_key, result_binary)
            logger.info(f"[缓存] ✓ {result_type}保存成功: {self.result_cache_bucket}/{result_key}")
        except Exception as e:
            logger.error(f"[缓存] 保存缓存失败: MD5={md5}, OCR类型={self.ocr_type}, 错误: {e}", exc_info=True)
            # 不抛出异常，允许继续处理
    
    def _call_ocr_api(self, pdf_binary: bytes) -> bytes:
        """调用OCR API解析PDF"""
        if not self.ocr_client:
            raise ValueError("OCR client not initialized. Please configure TEXTIN_OCR_APP_ID and TEXTIN_OCR_SECRET_CODE or PADDLE_OCR_API_URL")
        
        logger.info(f"[OCR API] 调用 {self.ocr_type} OCR API 解析PDF，文件大小: {len(pdf_binary)} bytes, API URL: {self.ocr_client.api_url}")
        import time
        ocr_start = time.time()
        
        try:
            if self.ocr_type == "textin":
                # Textin API 调用（保持原有逻辑）
                response_text = self.ocr_client.recognize(pdf_binary)
                ocr_duration = time.time() - ocr_start
                logger.info(f"[OCR API] Textin OCR API调用成功，耗时: {ocr_duration:.2f}秒，响应大小: {len(response_text)} bytes")
                
                # 解析响应，提取result字段
                json_response = json.loads(response_text)
                if "result" in json_response:
                    result_json = json.dumps(json_response["result"], ensure_ascii=False).encode('utf-8')
                    logger.info(f"[OCR API] 提取result字段成功，result大小: {len(result_json)} bytes")
                    return result_json
                else:
                    logger.error(f"[OCR API] Textin OCR API响应缺少'result'字段，响应键: {list(json_response.keys())}")
                    raise ValueError("Textin OCR API response missing 'result' field")
            
            elif self.ocr_type == "paddleocr":
                # PaddleOCR API 调用，返回 markdown 文本
                markdown_text = self.ocr_client.recognize(pdf_binary, options={"visualize": False})
                ocr_duration = time.time() - ocr_start
                logger.info(f"[OCR API] PaddleOCR 调用成功，耗时: {ocr_duration:.2f}秒，markdown大小: {len(markdown_text)} 字符")
                # 返回 markdown 文本的字节形式（用于缓存）
                return markdown_text.encode('utf-8')
            else:
                raise ValueError(f"Unknown OCR type: {self.ocr_type}")
                
        except requests.exceptions.RequestException as e:
            ocr_duration = time.time() - ocr_start
            logger.error(f"[OCR API] OCR API 调用失败（网络错误），耗时: {ocr_duration:.2f}秒，错误: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            ocr_duration = time.time() - ocr_start
            logger.error(f"[OCR API] OCR API 调用失败，耗时: {ocr_duration:.2f}秒，错误: {str(e)}", exc_info=True)
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
        自定义解析逻辑 - 支持PDF、JSON和Markdown文件
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
        elif filename.lower().endswith(('.md', '.markdown')):
            logger.info(f"[解析入口] 识别为Markdown文件，调用Markdown解析流程")
            markdown_content = binary.decode('utf-8')
            return self._parse_markdown_file(filename, markdown_content, **kwargs)
        else:
            # 对于不支持的文件类型，返回空列表
            logger.warning(f"[解析入口] Custom parser仅支持PDF、JSON和Markdown文件，当前文件: {filename} (扩展名: {file_ext})")
            return []
    
    def _parse_pdf_file(self, filename: str, binary: bytes, **kwargs) -> List[Dict[str, Any]]:
        """
        解析PDF文件 - 使用OCR API或缓存
        支持 Textin (JSON) 和 PaddleOCR (Markdown)
        """
        try:
            logger.info(f"[PDF解析] 开始解析PDF文件: {filename}, 文件大小: {len(binary)} bytes")
            
            # 1. 计算PDF的MD5值
            md5 = self._calculate_pdf_md5(binary)
            logger.info(f"[PDF解析] PDF文件MD5: {md5}")
            
            # 2. 检查缓存
            logger.info(f"[PDF解析] 检查缓存 (bucket: {self.result_cache_bucket}, OCR类型: {self.ocr_type})")
            cached_result = self._get_cached_result(md5)
            if cached_result:
                logger.info(f"[PDF解析] ✓ 缓存命中，使用缓存结果: {filename} (MD5: {md5}), 结果大小: {len(cached_result)} bytes")
                
                # 根据 OCR 类型解析缓存
                if self.ocr_type == "paddleocr":
                    # PaddleOCR 缓存的是 markdown 文本
                    markdown_content = cached_result.decode('utf-8')
                    chunks = self._parse_markdown_file(filename, markdown_content, **kwargs)
                else:
                    # Textin 缓存的是 JSON
                    chunks = self._parse_json_file(filename, cached_result, **kwargs)
                
                logger.info(f"[PDF解析] 缓存结果解析完成，生成 {len(chunks)} 个chunks")
                return chunks
            
            logger.info(f"[PDF解析] ✗ 缓存未命中，需要调用OCR API")
            
            # 3. 调用OCR API
            if not self.ocr_client:
                logger.error(f"[PDF解析] OCR客户端未初始化，无法解析PDF: {filename}")
                return []
            
            logger.info(f"[PDF解析] 调用 {self.ocr_type} OCR API 解析PDF: {filename} (MD5: {md5})")
            import time
            ocr_start_time = time.time()
            result_binary = self._call_ocr_api(binary)
            ocr_duration = time.time() - ocr_start_time
            logger.info(f"[PDF解析] OCR API 调用完成，耗时: {ocr_duration:.2f}秒，返回结果大小: {len(result_binary)} bytes")
            
            # 4. 保存到缓存
            logger.info(f"[PDF解析] 保存结果到缓存 (bucket: {self.result_cache_bucket}, OCR类型: {self.ocr_type})")
            self._save_to_cache(md5, binary, result_binary)
            logger.info(f"[PDF解析] ✓ 缓存保存完成")
            
            # 5. 解析结果
            logger.info(f"[PDF解析] 开始解析OCR返回的结果")
            if self.ocr_type == "paddleocr":
                # PaddleOCR 返回的是 markdown 文本
                markdown_content = result_binary.decode('utf-8')
                chunks = self._parse_markdown_file(filename, markdown_content, **kwargs)
            else:
                # Textin 返回的是 JSON
                chunks = self._parse_json_file(filename, result_binary, **kwargs)
            
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
    
    def _parse_markdown_file(self, filename: str, markdown_content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        解析 Markdown 格式文件（PaddleOCR 输出）
        基于 gen_title_report.py 的逻辑处理标题层级
        """
        try:
            logger.info(f"[Markdown解析] 开始解析Markdown文件: {filename}, 内容大小: {len(markdown_content)} 字符")
            
            # 重置章节状态
            self.title_hierarchy = []
            self.current_hierarchy = []
            logger.info(f"[Markdown解析] 章节层级状态已重置")
            
            # 使用 gen_title_report.py 的逻辑提取标题
            titles = extract_titles_from_markdown(markdown_content)
            logger.info(f"[Markdown解析] 提取了 {len(titles)} 个标题")
            
            # 调整标题层级（基于标题模式优先级）
            titles = adjust_title_levels(titles)
            logger.info(f"[Markdown解析] 标题层级调整完成")
            
            # 构建标题层级映射（用于后续处理）
            title_map = {}  # {line_num: title_info}
            for title in titles:
                line_num = title.get('line', 0)
                title_map[line_num] = title
            
            # 解析 markdown 内容，生成 chunks
            chunks = []
            lines = markdown_content.split('\n')
            current_section_content = []
            current_section_title_hierarchy = []
            current_page = 1  # 简化处理，假设从第1页开始
            
            for line_num, line in enumerate(lines, 1):
                # 检查是否是标题行
                if re.match(r"^#{1,6}\s+.*$", line):
                    match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                    if match:
                        title_text = match.group(2).strip()
                        
                        # 过滤无效标题
                        if title_text.startswith('<!--') or title_text.endswith('-->'):
                            continue
                        if not title_text or not is_valid_title(title_text):
                            continue
                        
                        # 如果当前 section 有内容，先保存为 chunk
                        if current_section_content:
                            chunk = self._create_chunk_from_section(
                                filename, 
                                current_section_title_hierarchy, 
                                current_section_content,
                                current_page
                            )
                            if chunk:
                                chunks.append(chunk)
                            current_section_content = []
                        
                        # 更新当前 section 的标题层级
                        title_info = title_map.get(line_num)
                        if title_info:
                            level = title_info.get('level', 2)
                            # 根据层级更新 current_hierarchy
                            # level 从 2 开始（对应 markdown 的 ##），所以需要减 2
                            hierarchy_level = max(0, level - 2)
                            # 删除超出当前层级的标题
                            while len(current_section_title_hierarchy) > hierarchy_level:
                                current_section_title_hierarchy.pop()
                            # 现在列表长度应该 <= hierarchy_level
                            # 如果层级等于列表长度，追加新标题
                            if hierarchy_level == len(current_section_title_hierarchy):
                                current_section_title_hierarchy.append(title_text)
                            elif hierarchy_level < len(current_section_title_hierarchy):
                                # 替换对应层级的标题（这种情况理论上不应该发生，因为 while 循环已经处理了）
                                # 但为了安全起见，还是处理一下
                                current_section_title_hierarchy[hierarchy_level] = title_text
                                # 删除超出层级的元素
                                while len(current_section_title_hierarchy) > hierarchy_level + 1:
                                    current_section_title_hierarchy.pop()
                            else:
                                # 如果层级大于列表长度，先扩展列表到 hierarchy_level 长度，然后追加
                                # 这种情况可能发生在第一个标题的层级不是0时
                                while len(current_section_title_hierarchy) < hierarchy_level:
                                    current_section_title_hierarchy.append("")
                                current_section_title_hierarchy.append(title_text)
                            self.title_hierarchy = current_section_title_hierarchy.copy()
                else:
                    # 普通文本行，添加到当前 section
                    if line.strip():
                        current_section_content.append(line)
            
            # 处理最后一个 section
            if current_section_content:
                chunk = self._create_chunk_from_section(
                    filename,
                    current_section_title_hierarchy,
                    current_section_content,
                    current_page
                )
                if chunk:
                    chunks.append(chunk)
            
            logger.info(f"[Markdown解析] Markdown文件 {filename} 解析完成，生成 {len(chunks)} 个chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"[Markdown解析] 处理Markdown文件 {filename} 时发生错误: {str(e)}", exc_info=True)
            return []
    
    def _create_chunk_from_section(self, filename: str, title_hierarchy: List[str], content_lines: List[str], page_id: int) -> Optional[Dict[str, Any]]:
        """
        从 section 创建 chunk
        标题按照层级拼入 chunk，叶子章节为一个 chunk 块
        """
        if not content_lines:
            return None
        
        # 过滤掉包含"参考文献"的标题层级（支持简体字和繁体字）
        if title_hierarchy:
            # 检查标题中是否包含"参考文献"（简体或繁体）
            # 去掉空格后比较，支持"參 考 文 献"、"参考文献"等格式
            for title in title_hierarchy:
                title_no_spaces = title.replace(' ', '').replace('　', '')
                if '参考文献' in title_no_spaces or '參考文献' in title_no_spaces:
                    return None
        
        # 创建基础文档结构
        doc = self._create_base_doc(filename)
        chunk = doc.copy()
        
        # 构建 section_title（包含文档名和标题层级）
        doc_name = re.sub(r"\.[a-zA-Z]+$", "", filename)
        if title_hierarchy:
            section_title = f"{doc_name} > {' > '.join(title_hierarchy)}"
        else:
            section_title = doc_name
        
        # 合并内容行
        text_content = '\n'.join(content_lines).strip()
        if not text_content:
            return None
        
        # 构建包含章节标题的完整内容
        if section_title:
            content_with_weight = f"[{section_title}]\n{text_content}"
        else:
            content_with_weight = text_content
        
        # 添加位置信息（简化处理，使用默认值）
        chunk["position_int"] = [[page_id, 0, 0, 0, 0]]
        chunk["page_num_int"] = [page_id]
        chunk["top_int"] = [0]
        
        chunk.update({
            "content_with_weight": content_with_weight,
            "section_title": section_title,
            "doc_type_kwd": "text"
        })
        
        # 使用RAGFlow标准分词
        tokenize(chunk, text_content, False)  # 假设是中文文档
        
        # 为包含章节标题的文本内容添加重要关键词字段
        if section_title:
            content_for_keywords = f"{section_title}"
            chunk["important_kwd"], chunk["important_tks"] = self._generate_keywords(
                content_for_keywords, topn=5, context=f"text_content '{section_title}'"
            )
        
        return chunk
    
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
        # result_cache_bucket 会根据 OCR 类型自动设置为 paddleocr-cache 或 textin-cache，不需要配置
        
        logger.info(f"[解析器初始化] Parser config: {parser_config}")
        logger.info(f"[解析器初始化] Custom config bucket设置: pdf_cache_bucket={custom_config.get('pdf_cache_bucket', 'pdf-cache')}, result_cache_bucket将根据OCR类型自动选择")
        
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
