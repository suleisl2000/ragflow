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
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
from pathlib import Path
from io import BytesIO
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, image saving will be disabled")

from rag.nlp import rag_tokenizer, tokenize, add_positions, tokenize_chunks, bullets_category, title_frequency
from api.db import ParserType, LLMType
from api.db.services.llm_service import LLMBundle         
from rag.prompts.generator import keyword_extraction
from rag.utils.storage_factory import STORAGE_IMPL
from api.utils.configs import read_config, get_base_config

logger = logging.getLogger(__name__)

# 扩展的标题模式列表（按优先级从高到低排序）
# 格式: (pattern, priority, dot_count)
# priority: 优先级，数字越小优先级越高
# dot_count: 点号数量（仅用于数字x.x格式，其他模式为None）
# 在相同 markdown 层级内，根据优先级动态映射到连续层级
TITLE_PATTERNS = [
    # 模式0: 以1个或多个空格开头的markdown标题，或作为兜底匹配所有不匹配其他模式的标题（优先级0，与模式1a同级）
    (
        r"^.+",
        0,
        None
    ),
    # 模式1a: 第X编/部分/篇/章（优先级0，最高）
    (
        r"^第[零一二三四五六七八九十百千0-9]+(分?编|部分|篇|章)",
        0,
        None
    ),
    # 模式1b: 第X节（优先级1）
    (
        r"^第[零一二三四五六七八九十百千0-9]+节",
        1,
        None
    ),
    # 模式1c: 第X条（优先级2）
    (
        r"^第[零一二三四五六七八九十百千0-9]+条",
        2,
        None
    ),
    # 模式2: 一、二、三、（中文数字+顿号/全角点号/空格，优先级3）
    # 支持：一、 一． 一 （中文数字后跟顿号、全角点号或空格）
    (
        r"^([零一二三四五六七八九十百千]+|[一二三四五六七八九十]+)([、．]|\s+)",
        3,
        None
    ),
    # 模式3: （一）（二）（中文括号+中文数字，优先级4）
    (
        r"^[（(]([零一二三四五六七八九十百]+|[一二三四五六七八九十]+)[）)]",
        4,
        None
    ),
    # 模式4: "数字+空格"、"数字+、"、"数字+."（优先级5）
    # 注意：不能匹配数字x.x.x格式（由模式5-9处理）
    # 注意：不能匹配"数字+空格+右括号"格式（由模式9处理）
    # 匹配格式：数字+空格/点号/顿号+标题内容
    # 负向前瞻：排除"数字+空格+1-3位数字"（短数字，如"2 10个"），但允许"数字+空格+4位数字"（年份，如"2 2019"）
    # 支持的格式示例："3 局限性和未来的方向"、"2 2019 年更新共识的主要内容"、"1. 适应证"、"1、 病史"
    (
        r"^([0-9]{1,2})(\s+|[、.．]\s*)(?![0-9]{1,3}(?![0-9])|[）)]|$)",
        5,
        None
    ),
    # 模式5: 数字x.x格式（优先级6）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        6,
        1
    ),
    # 模式6: 数字x.x.x格式（优先级7）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        7,
        2
    ),
    # 模式7: 数字x.x.x.x格式（优先级8）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        8,
        3
    ),
    # 模式8: 数字x.x.x.x.x格式（优先级9）
    (
        r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})(?![0-9.])",
        9,
        4
    ),
    # 模式9: 数字（1）格式或数字+空格+右括号格式（优先级10）
    # 支持: "（1）"、"(1)"、"1 ）"、"2 ）" 等格式
    (
        r"^(?:[（(]([0-9]{1,2})[）)]|([0-9]{1,2})\s+[）)])",
        10,
        None
    ),
    # 模式10: "问题1"、"问题一"、"临床问题1"、"临床问题 1"、"问题 1"、"陈述1"、"陈述 1"格式（优先级11）
    # 支持: "问题1"、"问题一"、"问题 1"、"临床问题1"、"临床问题 1"、"临床问题一"、"陈述1"、"陈述 1"、"陈述一" 等格式
    (
        r"^((?:临床)?(?:问题|陈述))\s*([0-9]+|[零一二三四五六七八九十百千]+)",
        11,
        None
    ),
    # 模式11: "推荐意见1"、"推荐意见一"格式（优先级12，低于模式10）
    # 支持: "推荐意见1"、"推荐意见一"、"推荐意见2"、"推荐意见二" 等格式
    # 注意：优先级低于模式10（问题1），由于正则表达式互斥（模式10以"问题"开头，模式11以"推荐意见"开头），不会同时匹配
    (
        r"^推荐意见([0-9]+|[零一二三四五六七八九十百千]+)",
        12,
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
    # 跳过模式0（索引0），从模式1a开始检查
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


def adjust_title_levels(titles: list) -> tuple:
    """
    调整标题层级：完全按照标题模式优先级确定绝对Level，不考虑markdown层级
    
    Args:
        titles: 标题列表，每个元素包含 'content', 'level', 'line', 'file'
    
    Returns:
        (adjusted_titles, has_conflict) 元组
        adjusted_titles: 调整后的标题列表
        has_conflict: 是否存在模式0和模式1在同一Level的冲突
    """
    if not titles:
        return titles, False
    
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
        return adjusted_titles, False
    
    # 对于模式0，只保留紧挨在模式2（"一、"）之前的匹配标题
    # 如果有多个"一、"标题，每个"一、"之前如果有匹配模式0的标题都要保留
    # 但是"二、"或其他模式2之前的模式0不保留
    # 注意：模式1a、1b、1c的正则表达式（用于判断是否匹配模式1）
    pattern1a_regex = TITLE_PATTERNS[1][0]  # 模式1a的正则表达式（第X编/部分/篇/章）
    pattern1b_regex = TITLE_PATTERNS[2][0]  # 模式1b的正则表达式（第X节）
    pattern1c_regex = TITLE_PATTERNS[3][0]  # 模式1c的正则表达式（第X条）
    pattern2_regex = TITLE_PATTERNS[4][0]  # 模式2的正则表达式（"一、"、"二、"等）
    
    # 判断是否匹配模式1（1a、1b、1c中的任意一个）
    def matches_pattern1(content):
        return (re.match(pattern1a_regex, content) or 
                re.match(pattern1b_regex, content) or 
                re.match(pattern1c_regex, content))
    
    # 找到所有模式2（"一、"）的标题位置
    keep_pattern0_indices = set()
    for i, (idx, title, priority, dot_count) in enumerate(valid_titles):
        if priority == 3:  # 模式2的优先级是3（已从1调整为3）
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
                        if prev_priority == 0 and not matches_pattern1(prev_title['content'].strip()):
                            # 标记为保留
                            keep_pattern0_indices.add(prev_idx)
    
    # 检查文档中是否只有模式0标题（用于决定是否保留所有模式0标题）
    # 支持的文档例子：如"2023年左心瓣膜术后三尖瓣反流诊疗中国专家共识"，只有模式0标题（如"发病机制"、"术前评估"等）
    has_other_patterns = False
    for idx, title, priority, dot_count in valid_titles:
        if priority > 0:  # 存在其他模式的标题（优先级 > 0）
            has_other_patterns = True
            break
        elif priority == 0:
            # 检查是否是模式1（模式1a/1b/1c的优先级也是0，但通过matches_pattern1区分）
            if matches_pattern1(title['content'].strip()):
                has_other_patterns = True
                break
    has_only_pattern0 = not has_other_patterns
    
    # 过滤掉所有未标记保留的模式0标题
    # 注意：前言性标题（匹配PREFACE_TITLES的标题）应该保留，不应该被过滤
    # 如果文档中只有模式0标题，则保留所有模式0标题（兼容只有模式0标题的文档场景）
    filtered_titles = []
    for idx, title, priority, dot_count in valid_titles:
        # 如果是模式0（优先级0且不匹配模式1）
        if priority == 0 and not matches_pattern1(title['content'].strip()):
            # 检查是否是前言性标题
            content = title['content'].strip()
            content_no_spaces = content.replace(' ', '').replace('　', '')
            content_no_spaces_no_colon = content_no_spaces.rstrip('：:')
            is_preface = (content in PREFACE_TITLES or 
                         content.rstrip('：:') in PREFACE_TITLES or
                         content_no_spaces in PREFACE_TITLES or
                         content_no_spaces_no_colon in PREFACE_TITLES)
            
            # 如果文档中只有模式0标题，保留所有模式0标题
            # 否则，只保留前言性标题或紧挨在"一、"之前的模式0标题（原有逻辑）
            if has_only_pattern0:
                # 保留所有模式0标题，不进行过滤
                pass
            else:
                # 原有逻辑：如果是前言性标题，保留；否则只有当它被标记为保留时（即紧挨在"一、"之前），才保留
                if not is_preface and idx not in keep_pattern0_indices:
                    continue
        filtered_titles.append((idx, title, priority, dot_count))
    valid_titles = filtered_titles
    
    # 检查模式0和模式1是否在同一Level（互斥检查）
    has_conflict = False
    has_pattern0 = False
    has_pattern1 = False
    
    for idx, title, priority, dot_count in valid_titles:
        if priority == 0:
            if matches_pattern1(title['content'].strip()):
                has_pattern1 = True
            else:
                has_pattern0 = True
    
    # 如果同时存在模式0和模式1，则冲突
    if has_pattern0 and has_pattern1:
        has_conflict = True
    
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
    
    return filtered_adjusted_titles, has_conflict


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
                # 图片路径保持原样：markdown/imgs/（不需要修正）
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
        
        # 短段落合并配置（复用parser_config中的chunk_token_num）
        parser_config = kwargs.get("parser_config", {})
        self.chunk_token_num = parser_config.get("chunk_token_num", 512)
        self.delimiter = parser_config.get("delimiter", r"\n")
        
        # 短段落合并配置参数（医疗指南场景最佳默认值）
        self.enable_short_paragraph_merge = self.custom_config.get("enable_short_paragraph_merge", True)
        short_paragraph_threshold_ratio = self.custom_config.get("short_paragraph_threshold_ratio", 0.25)  # 医疗指南场景：0.25
        self.short_paragraph_threshold = int(self.chunk_token_num * short_paragraph_threshold_ratio)
        
        max_merged_paragraph_length_ratio = self.custom_config.get("max_merged_paragraph_length_ratio", 1.0)
        self.max_merged_paragraph_length = int(self.chunk_token_num * max_merged_paragraph_length_ratio)
        
        # 短段落识别模式（如果未提供，使用内置模式）
        self.short_paragraph_patterns = self.custom_config.get("short_paragraph_patterns", None)
        if self.short_paragraph_patterns is None:
            # 内置默认模式
            self.short_paragraph_patterns = [
                r"^\([0-9]+\)",           # (1), (2), (10)
                r"^[0-9]+[\.、]",         # 1., 2., 1、, 2、
                r"^[（(][零一二三四五六七八九十百千0-9]+[）)]",  # （一）, (一)
                r"^[零一二三四五六七八九十百千]+[、．]",  # 一、, 二．
                r"^[①②③④⑤⑥⑦⑧⑨⑩]",  # 圆圈数字
                r"^[a-zA-Z][\.、\)]",     # a., b), A、, B)
            ]
        
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
        
        # 缓存bucket配置初始化（支持环境变量和配置文件）
        self._init_bucket_config()
        logger.info(f"[解析器初始化] 缓存bucket配置: PDF={self.pdf_cache_bucket}, Result={self.result_cache_bucket} (OCR类型: {self.ocr_type})")
    
    def _init_bucket_config(self):
        """初始化bucket配置，支持环境变量和配置文件"""
        # 从环境变量或配置文件读取bucket配置
        # 优先级：环境变量 > custom_config > 配置文件 > 默认值
        
        # PDF缓存bucket
        self.pdf_cache_bucket = (
            os.environ.get("PDF_CACHE_BUCKET") or
            self.custom_config.get("pdf_cache_bucket") or
            get_base_config("pdf_cache_bucket") or
            "pdf-cache"
        )
        
        # 初始化所有OCR类型的bucket配置（用于自适应缓存查找）
        self.textin_cache_bucket = (
            os.environ.get("TEXTIN_CACHE_BUCKET") or
            self.custom_config.get("textin_cache_bucket") or
            get_base_config("textin_cache_bucket") or
            "textin-cache"
        )
        self.paddleocr_cache_bucket = (
            os.environ.get("PADDLEOCR_CACHE_BUCKET") or
            self.custom_config.get("paddleocr_cache_bucket") or
            get_base_config("paddleocr_cache_bucket") or
            "paddleocr-cache"
        )
        
        # 根据OCR类型确定结果缓存bucket（用于保存缓存）
        if self.ocr_type == "paddleocr":
            self.result_cache_bucket = self.paddleocr_cache_bucket
        elif self.ocr_type == "textin":
            self.result_cache_bucket = self.textin_cache_bucket
        else:
            # 默认使用 textin-cache（向后兼容）
            self.result_cache_bucket = self.textin_cache_bucket
    
    def _init_llm_model(self):
        """初始化LLM模型用于关键词生成（索引构建时）"""
        try:
            # 获取租户ID
            tenant_id = self.custom_config.get("tenant_id")
            if tenant_id:
                # 从配置文件读取知识库构建时的关键词提取模型
                from api.utils.configs import get_base_config
                kb_llm_config = get_base_config("kb_default_llm", {}) or {}
                kb_default_models = kb_llm_config.get("default_models", {}) or {}
                chat_model = kb_default_models.get("chat_model", {}) or {}
                
                if chat_model.get("name"):
                    # 使用配置的知识库构建关键词提取模型
                    model_name = chat_model.get("name")
                    factory = chat_model.get("factory", "LocalAI")
                    model_id = f"{model_name}@{factory}"
                    self.chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_name=model_id)
                    logger.info(f"Custom parser LLM model initialized for KB indexing keyword extraction: {model_id} (tenant: {tenant_id})")
                else:
                    # 回退到默认模型
                    self.chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
                    logger.info(f"Custom parser LLM model initialized with default model (tenant: {tenant_id})")
            else:
                logger.warning("No tenant_id provided, keyword generation will be disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM model for keyword generation: {e}")
            self.chat_mdl = None
    
    def _generate_keywords(self, content: str, topn: int = 8, context: str = "") -> tuple:
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
            logger.debug(f"Cache hit for keywords: {context}")
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
    
    def _generate_keywords_impl(self, content: str, topn: int = 8, context: str = "") -> tuple:
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
        """初始化OCR配置，默认使用 PaddleOCR，支持 Textin 作为备选，优先级：custom_config.ocr_type > 环境变量 > 配置文件 > custom_config"""
        try:
            # 优先从 custom_config 中读取 ocr_type（用于 JSON 解析，不需要 OCR 客户端）
            explicit_ocr_type = self.custom_config.get('ocr_type')
            if explicit_ocr_type:
                self.ocr_type = explicit_ocr_type
                logger.info(f"OCR type set from custom_config: {explicit_ocr_type}")
                # 如果指定了 ocr_type，仍然尝试初始化 OCR 客户端（用于 PDF 解析）
                if explicit_ocr_type == "paddleocr":
                    paddle_api_url = (
                        os.getenv('PADDLE_OCR_API_URL') or
                        self.custom_config.get('paddle_ocr_api_url') or
                        "http://localhost:8080/layout-parsing"
                    )
                    try:
                        self.ocr_client = PaddleOCRClient(paddle_api_url)
                        logger.info(f"PaddleOCR client initialized with API URL: {paddle_api_url}")
                    except Exception as e:
                        logger.warning(f"PaddleOCR client initialization failed (will use for JSON parsing only): {e}")
                        self.ocr_client = None
                elif explicit_ocr_type == "textin":
                    # Textin OCR 配置
                    app_id = (
                        os.getenv('TEXTIN_OCR_APP_ID') or
                        self.custom_config.get('ocr_app_id')
                    )
                    secret_code = (
                        os.getenv('TEXTIN_OCR_SECRET_CODE') or
                        self.custom_config.get('ocr_secret_code')
                    )
                    if app_id and secret_code:
                        try:
                            self.ocr_client = TextinOCRClient(app_id, secret_code)
                            logger.info(f"Textin OCR client initialized")
                        except Exception as e:
                            logger.warning(f"Textin OCR client initialization failed (will use for JSON parsing only): {e}")
                            self.ocr_client = None
                return
            
            # 如果没有明确指定 ocr_type，使用原有逻辑
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
    
    def _get_cached_result(self, md5: str) -> Optional[Tuple[bytes, str]]:
        """
        自适应查找OCR缓存，优先使用已有缓存（无论OCR类型）
        
        Returns:
            Optional[Tuple[bytes, str]]: (缓存数据, OCR类型) 或 None
        """
        try:
            # 1. 先查 Textin 缓存
            textin_key = f"{md5}/json/{md5}.json"
            logger.debug(f"[缓存] 检查Textin缓存: bucket={self.textin_cache_bucket}, key={textin_key}")
            if STORAGE_IMPL.obj_exist(self.textin_cache_bucket, textin_key):
                result_binary = STORAGE_IMPL.get(self.textin_cache_bucket, textin_key)
                logger.info(f"[缓存] ✓ Textin OCR结果缓存命中: MD5={md5}, 大小={len(result_binary)} bytes")
                return result_binary, "textin"
            
            # 2. 再查 PaddleOCR 缓存
            logger.debug(f"[缓存] 检查PaddleOCR缓存: bucket={self.paddleocr_cache_bucket}, MD5={md5}")
            all_pruned_results = []
            page_idx = 0
            while True:
                json_key = f"{md5}/json/{md5}_page{page_idx:03d}.json"
                if not STORAGE_IMPL.obj_exist(self.paddleocr_cache_bucket, json_key):
                    break
                json_binary = STORAGE_IMPL.get(self.paddleocr_cache_bucket, json_key)
                try:
                    pruned_result = json.loads(json_binary.decode('utf-8'))
                    all_pruned_results.append(pruned_result)
                except Exception as e:
                    logger.warning(f"[缓存] 解析PaddleOCR页面JSON失败: {json_key}, 错误: {e}")
                page_idx += 1
            
            if all_pruned_results:
                # 构建统一的 JSON 结构（与 _call_ocr_api 保持一致）
                merged_json = {
                    "prunedResult": {
                        "model_settings": all_pruned_results[0].get("model_settings", {}) if all_pruned_results else {},
                        "parsing_res_list": []
                    }
                }
                
                # 合并所有页面的 parsing_res_list，并添加页码信息
                for page_idx, pruned_result in enumerate(all_pruned_results):
                    parsing_res_list = pruned_result.get("parsing_res_list", [])
                    for block in parsing_res_list:
                        # 添加页码信息到每个 block
                        block_with_page = block.copy()
                        block_with_page["page_index"] = page_idx + 1  # 页码从1开始
                        merged_json["prunedResult"]["parsing_res_list"].append(block_with_page)
                
                result_binary = json.dumps(merged_json, ensure_ascii=False).encode('utf-8')
                logger.info(f"[缓存] ✓ PaddleOCR结果缓存命中: MD5={md5}, 大小={len(result_binary)} bytes, 共 {len(all_pruned_results)} 页")
                return result_binary, "paddleocr"
            
            # 3. 都未找到，返回None（将使用当前配置的OCR类型调用API）
            logger.debug(f"[缓存] ✗ OCR结果缓存未命中: MD5={md5}, 将使用当前配置的OCR类型({self.ocr_type})调用API")
            return None, None
            
        except Exception as e:
            logger.warning(f"[缓存] 获取OCR结果缓存失败: MD5={md5}, 错误: {e}", exc_info=True)
        return None, None
    
    def _save_to_cache(self, md5: str, pdf_binary: bytes, result_binary: bytes, full_response: Dict[str, Any] = None):
        """
        保存PDF和OCR结果到对象存储（按文档目录结构组织）
        
        Args:
            md5: PDF文件的MD5值
            pdf_binary: PDF文件二进制数据
            result_binary: OCR结果的主要数据（markdown或json）
            full_response: OCR API的完整响应（包含图片等数据）
        """
        try:
            # 保存PDF到pdf-cache bucket（保持原有结构）
            pdf_key = f"{md5}.pdf"
            logger.info(f"[缓存] 保存PDF到缓存: bucket={self.pdf_cache_bucket}, key={pdf_key}, 大小={len(pdf_binary)} bytes")
            try:
                result = STORAGE_IMPL.put(self.pdf_cache_bucket, pdf_key, pdf_binary)
                if result:
                    logger.info(f"[缓存] ✓ PDF保存成功: {self.pdf_cache_bucket}/{pdf_key}")
                else:
                    logger.warning(f"[缓存] ✗ PDF保存失败（返回False）: {self.pdf_cache_bucket}/{pdf_key}")
            except Exception as e:
                logger.exception(f"[缓存] ✗ PDF保存异常: {self.pdf_cache_bucket}/{pdf_key}, 错误: {e}")
                raise
            
            bucket = self.result_cache_bucket
            logger.info(f"[缓存] 保存OCR结果到缓存: bucket={bucket}, OCR类型={self.ocr_type}")
            
            if self.ocr_type == "paddleocr":
                # PaddleOCR: 按照 paddle_ocr.py 的目录结构保存每页的 JSON 文件
                # 目录结构：{md5}/json/{md5}_page{page_idx:03d}.json
                if full_response:
                    layout_results = full_response.get("result", {}).get("layoutParsingResults", [])
                    
                    # 保存每页的JSON文件
                    for page_idx, res in enumerate(layout_results):
                        # 保存prunedResult
                        pruned_result = res.get("prunedResult", {})
                        json_key = f"{md5}/json/{md5}_page{page_idx:03d}.json"
                        json_data = json.dumps(pruned_result, ensure_ascii=False, indent=2).encode('utf-8')
                        STORAGE_IMPL.put(bucket, json_key, json_data)
                        logger.debug(f"[缓存] ✓ JSON保存成功: {bucket}/{json_key}")
                        
                        # 保存完整结果
                        full_json_key = f"{md5}/json/{md5}_page{page_idx:03d}_full.json"
                        full_json_data = json.dumps(res, ensure_ascii=False, indent=2).encode('utf-8')
                        STORAGE_IMPL.put(bucket, full_json_key, full_json_data)
                        logger.debug(f"[缓存] ✓ 完整JSON保存成功: {bucket}/{full_json_key}")
                        
                        # 保存每页的markdown
                        markdown_data = res.get("markdown", {})
                        markdown_text = markdown_data.get("text", "")
                        if markdown_text:
                            page_md_key = f"{md5}/markdown/{md5}_page{page_idx:03d}.md"
                            STORAGE_IMPL.put(bucket, page_md_key, markdown_text.encode('utf-8'))
                            logger.debug(f"[缓存] ✓ 页面Markdown保存成功: {bucket}/{page_md_key}")
                        
                        # 保存markdown中的图片（保存到markdown/imgs/目录，保持原路径）
                        if PIL_AVAILABLE:
                            markdown_images = markdown_data.get("images", {})
                            for img_path, img_base64 in markdown_images.items():
                                try:
                                    img_bytes = base64.b64decode(img_base64)
                                    # 图片路径保持原样：markdown/imgs/xxx（不需要images前缀）
                                    # img_path通常是 "imgs/xxx.jpg" 格式
                                    img_key = f"{md5}/markdown/{img_path}"
                                    STORAGE_IMPL.put(bucket, img_key, img_bytes)
                                    logger.debug(f"[缓存] ✓ Markdown图片保存成功: {bucket}/{img_key}")
                                except Exception as e:
                                    logger.warning(f"[缓存] 保存Markdown图片失败 {img_path}: {e}")
                            
                            # 保存可视化图片（outputImages）
                            output_images = res.get("outputImages", {})
                            for img_type, img_base64 in output_images.items():
                                try:
                                    img_bytes = base64.b64decode(img_base64)
                                    img_key = f"{md5}/images/{img_type}/{md5}_page{page_idx:03d}_{img_type}.png"
                                    STORAGE_IMPL.put(bucket, img_key, img_bytes)
                                    logger.debug(f"[缓存] ✓ 可视化图片保存成功: {bucket}/{img_key}")
                                except Exception as e:
                                    logger.warning(f"[缓存] 保存可视化图片失败 {img_type}: {e}")
            
            else:
                # Textin: 保存JSON和图片
                json_key = f"{md5}/json/{md5}.json"
                logger.debug(f"[缓存] 保存JSON到缓存: bucket={bucket}, key={json_key}, 大小={len(result_binary)} bytes")
                STORAGE_IMPL.put(bucket, json_key, result_binary)
                logger.info(f"[缓存] ✓ JSON保存成功: {bucket}/{json_key}")
                
                # 如果有完整响应，提取markdown和图片
                if full_response and "result" in full_response:
                    result = full_response["result"]
                    
                    # 保存markdown（如果有）
                    if "markdown" in result:
                        markdown_content = result["markdown"]
                        if isinstance(markdown_content, str):
                            md_key = f"{md5}/markdown/{md5}.md"
                            STORAGE_IMPL.put(bucket, md_key, markdown_content.encode('utf-8'))
                            logger.debug(f"[缓存] ✓ Markdown保存成功: {bucket}/{md_key}")
                            
                            # 提取并下载markdown中的图片（如果包含URL）
                            # 注意：Textin返回的markdown可能包含图片URL，需要下载
                            # 这里先保存markdown，图片下载可以在后续处理中完成
                            
        except Exception as e:
            logger.error(f"[缓存] 保存缓存失败: MD5={md5}, OCR类型={self.ocr_type}, 错误: {e}", exc_info=True)
            # 不抛出异常，允许继续处理
    
    def _save_textin_markdown_images(self, md5: str, markdown_content: str, bucket: str):
        """
        从Textin的markdown中提取图片URL并下载保存到对象存储
        图片保存到 markdown/imgs/ 目录，与PaddleOCR统一
        
        Args:
            md5: 文档MD5值
            markdown_content: Markdown内容（可能包含图片URL）
            bucket: 存储bucket名称
        """
        try:
            # 匹配markdown中的图片URL: ![...](https://...)
            import re
            pattern = r'!\[([^\]]*)\]\((https://[^\s\)]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg))\)'
            matches = re.findall(pattern, markdown_content, re.IGNORECASE)
            
            if not matches:
                logger.debug(f"[缓存] Textin markdown中未找到图片URL")
                return
            
            logger.info(f"[缓存] 从Textin markdown中提取到 {len(matches)} 个图片URL")
            
            # 下载并保存图片
            for idx, (alt_text, img_url) in enumerate(matches):
                try:
                    # 下载图片
                    response = requests.get(img_url, timeout=30)
                    response.raise_for_status()
                    img_bytes = response.content
                    
                    # 生成图片文件名
                    from urllib.parse import urlparse
                    parsed_url = urlparse(img_url)
                    original_filename = os.path.basename(parsed_url.path)
                    if not original_filename or '.' not in original_filename:
                        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
                        original_filename = f"{url_hash}.jpg"
                    
                    # 保存图片到 {md5}/markdown/imgs/{filename}（与PaddleOCR统一）
                    img_key = f"{md5}/markdown/imgs/{original_filename}"
                    STORAGE_IMPL.put(bucket, img_key, img_bytes)
                    logger.debug(f"[缓存] ✓ Markdown图片保存成功: {bucket}/{img_key} ({len(img_bytes)} bytes)")
                    
                    # 更新markdown中的图片路径为相对路径 imgs/{filename}
                    # 注意：这里只记录，实际的markdown更新可以在读取时处理
                    
                except Exception as e:
                    logger.warning(f"[缓存] 下载Textin图片失败 {img_url}: {e}")
                    
        except Exception as e:
            logger.warning(f"[缓存] 处理Textin markdown图片失败: {e}")
    
    def _call_ocr_api(self, pdf_binary: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        调用OCR API解析PDF
        
        Returns:
            Tuple[bytes, Dict]: (主要结果数据, 完整响应数据)
            - 对于Textin: (result JSON bytes, 完整响应)
            - 对于PaddleOCR: (markdown bytes, 完整响应)
        """
        if not self.ocr_client:
            raise ValueError("OCR client not initialized. Please configure TEXTIN_OCR_APP_ID and TEXTIN_OCR_SECRET_CODE or PADDLE_OCR_API_URL")
        
        logger.info(f"[OCR API] 调用 {self.ocr_type} OCR API 解析PDF，文件大小: {len(pdf_binary)} bytes, API URL: {self.ocr_client.api_url}")
        import time
        ocr_start = time.time()
        
        try:
            if self.ocr_type == "textin":
                # Textin API 调用
                response_text = self.ocr_client.recognize(pdf_binary)
                ocr_duration = time.time() - ocr_start
                logger.info(f"[OCR API] Textin OCR API调用成功，耗时: {ocr_duration:.2f}秒，响应大小: {len(response_text)} bytes")
                
                # 解析响应
                json_response = json.loads(response_text)
                if "result" in json_response:
                    result_json = json.dumps(json_response["result"], ensure_ascii=False).encode('utf-8')
                    logger.info(f"[OCR API] 提取result字段成功，result大小: {len(result_json)} bytes")
                    return result_json, json_response
                else:
                    logger.error(f"[OCR API] Textin OCR API响应缺少'result'字段，响应键: {list(json_response.keys())}")
                    raise ValueError("Textin OCR API response missing 'result' field")
            
            elif self.ocr_type == "paddleocr":
                # PaddleOCR API 调用，返回 JSON 数据
                file_base64 = base64.b64encode(pdf_binary).decode('utf-8')
                
                # 判断文件类型：0=PDF, 1=图片（与paddle_ocr.py保持一致）
                # 这里传入的是PDF二进制，所以fileType应该是0
                file_type = 0
                
                payload = {
                    "file": file_base64,
                    "fileType": file_type,  # 使用变量，与paddle_ocr.py保持一致
                    "visualize": True  # 获取图片数据
                }
                
                response = requests.post(
                    self.ocr_client.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=3600
                )
                response.raise_for_status()
                result_data = response.json()
                
                if result_data.get("errorCode") != 0:
                    error_msg = result_data.get("errorMsg", "Unknown error")
                    raise ValueError(f"PaddleOCR API error: {error_msg}")
                
                # 提取 JSON 数据（prunedResult）
                layout_results = result_data.get("result", {}).get("layoutParsingResults", [])
                # 合并所有页面的 prunedResult
                all_pruned_results = []
                for res in layout_results:
                    pruned_result = res.get("prunedResult", {})
                    if pruned_result:
                        all_pruned_results.append(pruned_result)
                
                # 构建统一的 JSON 结构
                merged_json = {
                    "prunedResult": {
                        "model_settings": all_pruned_results[0].get("model_settings", {}) if all_pruned_results else {},
                        "parsing_res_list": []
                    }
                }
                
                # 合并所有页面的 parsing_res_list，并添加页码信息
                for page_idx, pruned_result in enumerate(all_pruned_results):
                    parsing_res_list = pruned_result.get("parsing_res_list", [])
                    for block in parsing_res_list:
                        # 添加页码信息到每个 block
                        block_with_page = block.copy()
                        block_with_page["page_index"] = page_idx + 1  # 页码从1开始
                        merged_json["prunedResult"]["parsing_res_list"].append(block_with_page)
                
                result_json = json.dumps(merged_json, ensure_ascii=False).encode('utf-8')
                ocr_duration = time.time() - ocr_start
                logger.info(f"[OCR API] PaddleOCR 调用成功，耗时: {ocr_duration:.2f}秒，JSON大小: {len(result_json)} bytes，共 {len(all_pruned_results)} 页")
                
                # 返回 JSON 数据和完整响应
                return result_json, result_data
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
            # 判断是 Textin JSON 还是 PaddleOCR JSON
            # 根据 ocr_type 决定使用哪个解析方法
            if self.ocr_type == "paddleocr":
                return self._parse_paddleocr_json_file(filename, binary, **kwargs)
            else:
                return self._parse_json_file(filename, binary, **kwargs)
        else:
            # 对于不支持的文件类型，返回空列表
            logger.warning(f"[解析入口] Custom parser仅支持PDF和JSON文件，当前文件: {filename} (扩展名: {file_ext})")
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
            
            # 2. 自适应检查缓存（优先使用已有缓存，无论OCR类型）
            logger.info(f"[PDF解析] 自适应检查缓存 (Textin bucket: {self.textin_cache_bucket}, PaddleOCR bucket: {self.paddleocr_cache_bucket})")
            cached_result, cached_ocr_type = self._get_cached_result(md5)
            if cached_result and cached_ocr_type:
                logger.info(f"[PDF解析] ✓ 缓存命中，使用缓存结果: {filename} (MD5: {md5}), OCR类型: {cached_ocr_type}, 结果大小: {len(cached_result)} bytes")
                
                # 临时设置ocr_type以使用对应的解析方法
                original_ocr_type = self.ocr_type
                self.ocr_type = cached_ocr_type
                
                try:
                    # 根据缓存的 OCR 类型解析缓存
                    if cached_ocr_type == "paddleocr":
                        # PaddleOCR 缓存的是 JSON 数据
                        chunks = self._parse_paddleocr_json_file(filename, cached_result, **kwargs)
                    else:
                        # Textin 缓存的是 JSON
                        chunks = self._parse_json_file(filename, cached_result, **kwargs)
                    
                    logger.info(f"[PDF解析] 缓存结果解析完成，生成 {len(chunks)} 个chunks")
                    return chunks
                finally:
                    # 恢复原始ocr_type
                    self.ocr_type = original_ocr_type
            
            logger.info(f"[PDF解析] ✗ 缓存未命中，需要调用OCR API")
            
            # 3. 调用OCR API
            if not self.ocr_client:
                logger.error(f"[PDF解析] OCR客户端未初始化，无法解析PDF: {filename}")
                return []
            
            logger.info(f"[PDF解析] 调用 {self.ocr_type} OCR API 解析PDF: {filename} (MD5: {md5})")
            import time
            ocr_start_time = time.time()
            result_binary, full_response = self._call_ocr_api(binary)
            ocr_duration = time.time() - ocr_start_time
            logger.info(f"[PDF解析] OCR API 调用完成，耗时: {ocr_duration:.2f}秒，返回结果大小: {len(result_binary)} bytes")
            
            # 4. 保存到缓存（包括图片等完整数据）
            logger.info(f"[PDF解析] 保存结果到缓存 (bucket: {self.result_cache_bucket}, OCR类型: {self.ocr_type})")
            self._save_to_cache(md5, binary, result_binary, full_response)
            logger.info(f"[PDF解析] ✓ 缓存保存完成")
            
            # 5. 解析结果
            logger.info(f"[PDF解析] 开始解析OCR返回的结果")
            if self.ocr_type == "paddleocr":
                # PaddleOCR 返回的是 JSON 数据
                chunks = self._parse_paddleocr_json_file(filename, result_binary, **kwargs)
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
    
    def _parse_paddleocr_json_file(self, filename: str, json_binary: bytes, **kwargs) -> List[Dict[str, Any]]:
        """
        解析 PaddleOCR JSON 格式文件
        从 JSON 数据中提取标题层级和段落，生成段落级和章节级 chunks
        """
        try:
            logger.info(f"[PaddleOCR JSON解析] 开始解析JSON文件: {filename}, JSON大小: {len(json_binary)} bytes")
            
            # 解析JSON数据
            json_data = json.loads(json_binary.decode('utf-8'))
            logger.info(f"[PaddleOCR JSON解析] JSON解析成功，根对象类型: {type(json_data).__name__}")
            
            # 重置章节状态
            self.title_hierarchy = []
            self.current_hierarchy = []
            logger.info(f"[PaddleOCR JSON解析] 章节层级状态已重置")
            
            # 检查JSON格式
            if not isinstance(json_data, dict) or 'prunedResult' not in json_data:
                logger.warning(f"[PaddleOCR JSON解析] JSON文件 {filename} 格式不正确，缺少prunedResult字段")
                return []
            
            pruned_result = json_data.get('prunedResult', {})
            parsing_res_list = pruned_result.get('parsing_res_list', [])
            
            if not isinstance(parsing_res_list, list):
                logger.warning(f"[PaddleOCR JSON解析] JSON文件 {filename} 的parsing_res_list字段不是列表类型")
                return []
            
            # 从文件名提取页码并添加到每个 block（如果 block 没有 page_index）
            # 优先使用 block 中的 _source_filename（如果存在，说明是合并后的 JSON，需要从原始文件名提取）
            # 否则使用传入的 filename
            import re
            for block in parsing_res_list:
                if 'page_index' not in block or block.get('page_index', 0) == 0:
                    # 优先使用 block 中的 _source_filename（合并后的 JSON）
                    source_filename = block.get('_source_filename', filename)
                    page_match = re.search(r'_page(\d+)', source_filename)
                    if page_match:
                        page_index = int(page_match.group(1)) + 1  # 页码从1开始
                        block['page_index'] = page_index
                    else:
                        # 如果文件名中没有页码，使用默认值1
                        block['page_index'] = 1
                # 删除临时字段 _source_filename（不再需要）
                if '_source_filename' in block:
                    del block['_source_filename']
            
            logger.info(f"[PaddleOCR JSON解析] parsing_res_list包含 {len(parsing_res_list)} 个block")
            
            # 提取标题层级和段落
            paragraphs, sections = self._extract_paragraphs_from_json_blocks(parsing_res_list, filename, kwargs.get("doc_id", ""))
            logger.info(f"[PaddleOCR JSON解析] 提取了 {len(paragraphs)} 个段落，{len(sections)} 个章节")
            
            # 创建基础文档结构（用于获取 docnm_kwd 等字段）
            base_doc = self._create_base_doc(filename)
            
            # 创建 chunks
            chunks = []
            
            # 创建段落级 chunks（用于检索）
            for para in paragraphs:
                para_chunk = self._create_paragraph_chunk(para, filename, base_doc, **kwargs)
                if para_chunk:
                    chunks.append(para_chunk)
            
            # 创建章节级 chunks（用于返回）
            for section in sections:
                section_chunk = self._create_section_chunk(section, filename, base_doc, **kwargs)
                if section_chunk:
                    chunks.append(section_chunk)
            
            logger.info(f"[PaddleOCR JSON解析] JSON文件 {filename} 解析完成: 生成 {len(chunks)} 个chunks (段落: {len(paragraphs)}, 章节: {len(sections)})")
            return chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"[PaddleOCR JSON解析] JSON文件 {filename} 解析失败（JSON格式错误）: {str(e)}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"[PaddleOCR JSON解析] 处理JSON文件 {filename} 时发生错误: {str(e)}", exc_info=True)
            return []
    
    def _determine_title_level_dynamic(self, title_text: str, pattern_priority: int, dot_count: Optional[int], 
                                       pattern_to_level: Dict[Tuple[int, Optional[int]], int]) -> int:
        """
        动态判定标题的 level
        
        Args:
            title_text: 标题文本
            pattern_priority: 模式优先级
            dot_count: 点号数量（仅用于数字x.x格式）
            pattern_to_level: 模式到 level 的映射字典 {(priority, dot_count): level}
        
        Returns:
            标题的 level（从1开始）
        """
        pattern_key = (pattern_priority, dot_count)
        
        # 如果该模式已经分配过 level，直接返回
        if pattern_key in pattern_to_level:
            return pattern_to_level[pattern_key]
        
        # 找到当前已分配的最大 level
        # 注意：adjust_title_levels 返回的 level 从 2 开始（对应 markdown 的 ##），所以这里也要从 2 开始
        max_level = max(pattern_to_level.values()) if pattern_to_level else 1  # 如果为空，从 1 开始，下一个是 2
        
        # 新模式的 level = max_level + 1
        # 确保至少从 2 开始（与 adjust_title_levels 保持一致）
        new_level = max(max_level + 1, 2)
        pattern_to_level[pattern_key] = new_level
        
        logger.debug(f"[动态Level判定] 标题 '{title_text}' 模式 ({pattern_priority}, {dot_count}) 分配 level {new_level}")
        return new_level
    
    def _is_short_paragraph(self, block_content: str) -> bool:
        """
        判断是否为短段落
        
        Args:
            block_content: block的内容
        
        Returns:
            True表示是短段落，False表示不是
        """
        if not self.enable_short_paragraph_merge:
            return False
        
        content = block_content.strip()
        if not content:
            return False
        
        # 检查字符数：如果字符数小于阈值，认为是短段落
        if len(content) < self.short_paragraph_threshold:
            return True
        
        # 如果字符数超过阈值，但匹配编号列表模式，也认为是短段落（用于处理较长的编号列表项）
        for pattern in self.short_paragraph_patterns:
            if re.match(pattern, content):
                return True
        
        return False
    
    def _merge_short_paragraphs_in_batches(self, pending_short_paragraphs: List[Dict[str, Any]], 
                                           pending_short_paragraphs_page_index: Optional[int],
                                           current_section_path: List[str],
                                           get_parent_section_id,
                                           filename: str,
                                           paragraph_counter: int,
                                           paragraphs: List[Dict[str, Any]],
                                           current_section: Optional[Dict[str, Any]],
                                           current_section_paragraph_ids: List[str]) -> int:
        """
        分批合并短段落
        
        如果合并后超过限制，尝试分批合并，而不是分别创建。
        这样可以保持相关短段落的语义完整性。
        
        Args:
            pending_short_paragraphs: 待合并的短段落列表
            pending_short_paragraphs_page_index: 短段落列表的起始页码
            current_section_path: 当前章节路径
            get_parent_section_id: 获取父章节ID的函数
            filename: 文件名
            paragraph_counter: 当前段落计数器
            paragraphs: 段落列表
            current_section: 当前章节
            current_section_paragraph_ids: 当前章节的段落ID列表
        
        Returns:
            更新后的段落计数器
        """
        if not pending_short_paragraphs:
            return paragraph_counter
        
        # 先尝试一次性合并所有短段落
        merged_content_parts = [p["block_content"] for p in pending_short_paragraphs]
        merged_content = "\n".join(merged_content_parts)
        
        # 如果合并后不超过限制，直接合并
        if len(merged_content) <= self.max_merged_paragraph_length:
            paragraph_counter += 1
            para_chunk_id = f"{filename}_para_{paragraph_counter}"
            paragraph = {
                "content": merged_content,
                "page_index": pending_short_paragraphs_page_index or 1,
                "section_path": current_section_path.copy(),
                "parent_section_id": get_parent_section_id(para_chunk_id),
                "chunk_id": para_chunk_id
            }
            paragraphs.append(paragraph)
            if current_section:
                current_section_paragraph_ids.append(para_chunk_id)
        else:
            # 超过长度限制，尝试分批合并
            remaining_paragraphs = pending_short_paragraphs.copy()
            batch_page_index = pending_short_paragraphs_page_index
            
            while remaining_paragraphs:
                # 尝试合并尽可能多的短段落
                batch = []
                batch_content = ""
                
                for p in remaining_paragraphs:
                    # 尝试加入当前短段落
                    test_content = batch_content + "\n" + p["block_content"] if batch_content else p["block_content"]
                    
                    if len(test_content) <= self.max_merged_paragraph_length:
                        # 可以加入当前批次
                        batch.append(p)
                        batch_content = test_content
                    else:
                        # 超过限制，停止当前批次
                        break
                
                # 处理当前批次
                if len(batch) > 1:
                    # 多个短段落，合并创建
                    paragraph_counter += 1
                    para_chunk_id = f"{filename}_para_{paragraph_counter}"
                    paragraph = {
                        "content": batch_content,
                        "page_index": batch_page_index if batch_page_index is not None else (batch[0]["page_index"] if batch else 1),
                        "section_path": current_section_path.copy(),
                        "parent_section_id": get_parent_section_id(para_chunk_id),
                        "chunk_id": para_chunk_id
                    }
                    paragraphs.append(paragraph)
                    if current_section:
                        current_section_paragraph_ids.append(para_chunk_id)
                elif len(batch) == 1:
                    # 只有一个短段落，单独创建
                    paragraph_counter += 1
                    para_chunk_id = f"{filename}_para_{paragraph_counter}"
                    paragraph = {
                        "content": batch[0]["block_content"],
                        "page_index": batch[0]["page_index"],
                        "section_path": current_section_path.copy(),
                        "parent_section_id": get_parent_section_id(para_chunk_id),
                        "chunk_id": para_chunk_id
                    }
                    paragraphs.append(paragraph)
                    if current_section:
                        current_section_paragraph_ids.append(para_chunk_id)
                else:
                    # batch为空，说明第一个短段落就超过限制，单独处理它
                    # 这种情况不应该发生（因为短段落应该小于阈值），但为了安全起见还是处理
                    if remaining_paragraphs:
                        paragraph_counter += 1
                        para_chunk_id = f"{filename}_para_{paragraph_counter}"
                        paragraph = {
                            "content": remaining_paragraphs[0]["block_content"],
                            "page_index": remaining_paragraphs[0]["page_index"],
                            "section_path": current_section_path.copy(),
                            "parent_section_id": get_parent_section_id(para_chunk_id),
                            "chunk_id": para_chunk_id
                        }
                        paragraphs.append(paragraph)
                        if current_section:
                            current_section_paragraph_ids.append(para_chunk_id)
                        # 移除已处理的短段落
                        remaining_paragraphs = remaining_paragraphs[1:]
                        if remaining_paragraphs:
                            batch_page_index = remaining_paragraphs[0]["page_index"]
                        continue
                
                # 移除已处理的短段落
                remaining_paragraphs = remaining_paragraphs[len(batch):]
                # 更新批次页码（用于下一批次）
                if batch:
                    batch_page_index = batch[-1]["page_index"]
        
        return paragraph_counter
    
    def _extract_paragraphs_from_json_blocks(self, blocks: List[Dict[str, Any]], filename: str, doc_id: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从 JSON blocks 中提取段落和章节
        
        Args:
            blocks: parsing_res_list 中的 block 列表
            filename: 文件名
            doc_id: 文档ID，用于生成section_id和parent_section_id
        
        Returns:
            (paragraphs, sections) 元组
            - paragraphs: 段落列表，每个段落包含 content, page_index, section_path, parent_section_id 等
            - sections: 章节列表，每个章节包含 section_id, section_path, paragraph_chunk_ids, content 等
        """
        paragraphs = []
        sections = []
        
        # 第一遍：收集所有标题，用于统一分配层级（与 gen_title_report.py 保持一致）
        title_blocks = []
        # 建立 block 到 title_blocks 索引的映射（用于后续快速查找）
        block_to_title_idx: Dict[tuple, int] = {}  # (page_index, block_order, block_id, block_content) -> title_blocks index
        for idx, block in enumerate(blocks):
            block_label = block.get("block_label", "")
            # 保留标题内容中的前导空格（用于匹配模式0），只去掉尾随空格（与 gen_title_report.py 保持一致）
            block_content = block.get("block_content", "").rstrip()
            if block_label == "paragraph_title" and block_content:
                # 与 gen_title_report.py 保持一致：只使用 is_valid_title 过滤
                # 过滤掉空标题（已在上面检查）
                # 过滤掉没有匹配任何标题模式的标题（通过 is_valid_title）
                if is_valid_title(block_content):
                    # 计算行号：直接从 block 获取 page_index 和 block_order/block_id
                    # 每个 block 都有 page_index（在 _call_ocr_api 中已添加，或从 pagexxx 文件读取时已添加）
                    page_index = block.get("page_index", 1)
                    block_order = block.get("block_order")
                    block_id = block.get("block_id", 0)
                    # 使用 page_index * 10000 + block_order/block_id 作为行号
                    if block_order is not None:
                        line_num = page_index * 10000 + block_order
                    elif block_id > 0:
                        line_num = page_index * 10000 + block_id
                    else:
                        line_num = page_index * 10000 + 1
                    
                    title_idx = len(title_blocks)
                    title_blocks.append({
                        'content': block_content,
                        'level': 2,  # 临时值，会被 adjust_title_levels 覆盖
                        'line': line_num,  # 使用 page_index 和 block 顺序计算行号
                        'file': filename,
                        'block': block  # 保存原始 block 引用
                    })
                    # 建立映射：通过 (page_index, block_order, block_id, block_content) 来查找
                    block_key = (page_index, block_order, block_id, block_content)
                    block_to_title_idx[block_key] = title_idx
        
        # 使用 adjust_title_levels 统一分配层级（与 gen_title_report.py 保持一致）
        pattern_to_level: Dict[Tuple[int, Optional[int]], int] = {}
        # 创建保留的标题集合（用于过滤，与 gen_title_report.py 保持一致）
        # 只处理在 adjust_title_levels 返回的 adjusted_titles 中的标题
        # 使用索引集合来跟踪哪些 title_blocks 被保留了（更可靠）
        kept_title_indices: Set[int] = set()
        kept_titles_set: Set[str] = set()
        
        if title_blocks:
            adjusted_titles, has_conflict = adjust_title_levels(title_blocks)
            # 创建 (priority, dot_count) 到 level 的映射
            # 注意：adjust_title_levels 返回的 level 从 2 开始（对应 markdown 的 ##），
            # 我们直接使用这个 level，在路径构建时 level 2 对应路径深度 0，level 3 对应路径深度 1，以此类推
            for title_info in adjusted_titles:
                title_content = title_info['content']
                # 找到该标题在 title_blocks 中的索引（通过比较 content 和 line）
                # 注意：由于 adjust_title_levels 可能改变了 title 对象的引用，我们需要通过内容匹配
                for idx, orig_title in enumerate(title_blocks):
                    # 通过比较 content 和 line 来匹配（line 是唯一的）
                    if (orig_title['content'] == title_content and 
                        orig_title['line'] == title_info.get('line', 0)):
                        kept_title_indices.add(idx)
                        break
                
                # 添加到保留的标题集合（用于后续过滤）
                # 添加所有可能的变体，确保能匹配到（考虑前导空格、尾随空格、尾随冒号等）
                # 注意：title_content 来自 title_blocks，已经是 block_content.rstrip() 的结果
                kept_titles_set.add(title_content)
                kept_titles_set.add(title_content.strip())  # 去掉前后空格
                kept_titles_set.add(title_content.rstrip())  # 只去掉尾随空格（虽然理论上已经是 rstrip() 的结果）
                kept_titles_set.add(title_content.lstrip())  # 只去掉前导空格
                # 同时添加去掉尾随冒号的版本（因为可能有"要点提示："和"要点提示"两种形式）
                if title_content.endswith('：') or title_content.endswith(':'):
                    title_no_colon = title_content.rstrip('：:')
                    kept_titles_set.add(title_no_colon)
                    kept_titles_set.add(title_no_colon.strip())
                    kept_titles_set.add(title_no_colon.rstrip())
                    kept_titles_set.add(title_no_colon.lstrip())
                
                priority, dot_count = get_title_pattern_info(title_content)
                pattern_key = (priority, dot_count)
                # adjust_title_levels 返回的 level 从 2 开始（Level 2=##, Level 3=###, Level 4=####）
                # 在路径构建时，level 2 对应路径深度 0（顶级），level 3 对应路径深度 1，以此类推
                pattern_to_level[pattern_key] = title_info['level']
        
        # 当前章节路径（用于构建 section_path）
        current_section_path: List[str] = []
        
        # 当前章节信息
        current_section: Optional[Dict[str, Any]] = None
        current_section_paragraph_ids: List[str] = []
        
        # 跨页段落拼接：存储上一个页面的最后一个 text block
        last_text_block: Optional[Dict[str, Any]] = None
        last_page_index: Optional[int] = None
        
        # 短段落合并：暂存待合并的短段落
        pending_short_paragraphs: List[Dict[str, Any]] = []  # 存储待合并的短段落block
        pending_short_paragraphs_page_index: Optional[int] = None
        
        # 段落计数器（用于生成段落 chunk ID）
        paragraph_counter = 0
        
        # 标志：刚刚遇到了paragraph_title或doc_title（用于判断是否应该跨页合并）
        # 如果为True，说明当前text block和last_text_block之间有标题，不应该跨页合并
        just_encountered_title: bool = False
        
        # 辅助函数：获取parent_section_id并添加调试日志
        # 注意：使用xxhash方式生成id（section_path + doc_id），与section chunk的id生成方式一致
        def get_parent_section_id(para_chunk_id: str) -> str:
            """获取parent_section_id，如果存在问题则记录警告"""
            import xxhash
            parent_section_id = None
            if current_section:
                # 直接使用current_section中已生成的section_id，确保与section chunk的id完全一致
                parent_section_id = current_section.get("section_id", "")
                if not parent_section_id:
                    # Fallback: 如果section_id为空，重新计算（理论上不应该发生）
                    logger.warning(f"[段落提取] 警告: current_section存在但section_id为空! "
                                 f"filename={filename}, para_chunk_id={para_chunk_id}, "
                                 f"section_path={current_section_path}, current_section={current_section}")
                    section_path_str = " > ".join([title.strip() for title in current_section_path]) if current_section_path else ""
                    if section_path_str and doc_id:
                        parent_section_id = xxhash.xxh64((section_path_str + doc_id).encode("utf-8", "surrogatepass")).hexdigest()
                    elif section_path_str:
                        parent_section_id = section_path_str
            elif current_section_path:
                logger.warning(f"[段落提取] 警告: section_path存在但current_section为None! "
                             f"filename={filename}, para_chunk_id={para_chunk_id}, "
                             f"section_path={current_section_path}")
                # Fallback: 如果current_section是None但section_path存在，使用section_path + doc_id生成xxhash id
                normalized_section_path = [title.strip() for title in current_section_path]
                section_path_str = " > ".join(normalized_section_path)
                if section_path_str and doc_id:
                    parent_section_id = xxhash.xxh64((section_path_str + doc_id).encode("utf-8", "surrogatepass")).hexdigest()
                elif section_path_str:
                    # 如果没有doc_id，使用section_path作为临时值（理论上不应该发生）
                    parent_section_id = section_path_str
            return parent_section_id or ""
        
        for block in blocks:
            block_label = block.get("block_label", "")
            page_index = block.get("page_index", 1)
            block_id = block.get("block_id", 0)
            
            # 处理标题（paragraph_title）
            if block_label == "paragraph_title":
                # 对于标题，保留前导空格（用于匹配模式0），只去掉尾随空格（与 gen_title_report.py 保持一致）
                block_content = block.get("block_content", "").rstrip()
                
                # 跳过空内容
                if not block_content:
                    continue
                
                # 与 gen_title_report.py 保持一致：只使用 is_valid_title 过滤
                # 如果标题不匹配任何模式，跳过（通过 adjust_title_levels 会过滤掉）
                # 但这里需要先检查，确保只处理有效的标题
                if not is_valid_title(block_content):
                    continue
                
                # 检查该标题是否在 adjust_title_levels 返回的保留标题集合中
                # 如果不在，说明该标题在 adjust_title_levels 中被过滤掉了，应该跳过
                # 与 gen_title_report.py 保持一致：只处理经过 adjust_title_levels 过滤后的标题
                # 方法1：通过 block_key 查找 title_blocks 索引
                # 注意：使用与构建 block_key 时相同的变量，确保一致性
                block_order = block.get("block_order")
                block_key = (page_index, block_order, block_id, block_content)
                title_idx = block_to_title_idx.get(block_key)
                if title_idx is not None and title_idx not in kept_title_indices:
                    continue
                
                # 方法2：如果方法1找不到，使用内容匹配（备用方案）
                if title_idx is None:
                    block_content_variants = [
                        block_content,
                        block_content.strip(),
                        block_content.rstrip(),
                        block_content.lstrip(),
                    ]
                    # 如果 block_content 以冒号结尾，也检查去掉冒号的版本
                    if block_content.endswith('：') or block_content.endswith(':'):
                        block_content_variants.extend([
                            block_content.rstrip('：:'),
                            block_content.rstrip('：:').strip(),
                            block_content.rstrip('：:').rstrip(),
                            block_content.rstrip('：:').lstrip(),
                        ])
                    
                    # 如果所有变体都不在 kept_titles_set 中，说明该标题被过滤掉了
                    if not any(variant in kept_titles_set for variant in block_content_variants):
                        continue
                
                # ========== 修复：先处理last_text_block，避免误触发跨页合并 ==========
                # 如果遇到标题，先处理待合并的短段落（如果有）
                last_block_in_pending = False
                if last_text_block and pending_short_paragraphs:
                    last_block_content = last_text_block.get("block_content", "").strip()
                    last_block_in_pending = any(
                        p.get("block_content", "").strip() == last_block_content 
                        for p in pending_short_paragraphs
                    )
                
                if pending_short_paragraphs:
                    # 使用分批合并策略处理短段落
                    paragraph_counter = self._merge_short_paragraphs_in_batches(
                        pending_short_paragraphs,
                        pending_short_paragraphs_page_index,
                        current_section_path,
                        get_parent_section_id,
                        filename,
                        paragraph_counter,
                        paragraphs,
                        current_section,
                        current_section_paragraph_ids
                    )
                    pending_short_paragraphs = []
                    pending_short_paragraphs_page_index = None
                
                # 如果遇到标题，上一个 text block 应该单独成段
                # 注意：如果last_text_block已经在pending_short_paragraphs中被处理了，不应该重复处理
                if last_text_block and not last_block_in_pending:
                    paragraph_counter += 1
                    para_chunk_id = f"{filename}_para_{paragraph_counter}"
                    paragraph = {
                        "content": last_text_block.get("block_content", "").strip(),
                        "page_index": last_page_index or 1,
                        "section_path": current_section_path.copy(),
                        "parent_section_id": get_parent_section_id(para_chunk_id),
                        "chunk_id": para_chunk_id
                    }
                    paragraphs.append(paragraph)
                    if current_section:
                        current_section_paragraph_ids.append(para_chunk_id)
                
                # ========== 修复：无论last_text_block是否被处理，都要清空，避免在else分支中重复处理 ==========
                # 清空last_text_block，避免在else分支中重复处理
                last_text_block = None
                last_page_index = None
                # ========== 修复结束 ==========
                
                # 设置标志：刚刚遇到了标题
                just_encountered_title = True
                # ========== 修复结束 ==========
                
                # 处理当前标题
                # 获取标题的模式信息（block_content 已保留前导空格）
                priority, dot_count = get_title_pattern_info(block_content)
                
                if priority >= 0:  # 匹配到有效模式
                    # 使用统一分配的 level（与 gen_title_report.py 保持一致）
                    pattern_key = (priority, dot_count)
                    if pattern_key in pattern_to_level:
                        level = pattern_to_level[pattern_key]
                    else:
                        # 如果模式不在映射中（理论上不应该发生），使用动态分配作为后备
                        level = self._determine_title_level_dynamic(block_content, priority, dot_count, pattern_to_level)
                    
                    # 更新章节路径
                    # adjust_title_levels 返回的 level 从 2 开始（Level 2=##, Level 3=###, Level 4=####）
                    # 路径深度 = level - 2（Level 2 对应深度 0，Level 3 对应深度 1，以此类推）
                    path_depth = level - 2
                    
                    # 如果新标题的路径深度小于当前路径长度，需要截断路径
                    while len(current_section_path) > path_depth:
                        current_section_path.pop()
                    
                    # 判断是同级标题还是子级标题
                    # 关键：使用和"章节层级详情"完全一样的逻辑，直接从 pattern_to_level 获取路径中最后一个标题的 level
                    # 如果 level 相同，说明是同级标题，需要替换最后一个（不管 path_depth 是否等于 len(current_section_path)）
                    # 如果 level 不同，说明是子级标题或更高级的标题，需要追加或替换
                    # 注意：如果 path_depth < len(current_section_path)，说明是更高级的标题，已经通过上面的 while 循环截断了
                    last_title_level_in_path = None
                    if len(current_section_path) > 0:
                        # 从路径中最后一个标题获取其 level（使用和"章节层级详情"完全一样的逻辑）
                        last_title_content = current_section_path[-1]
                        last_priority, last_dot_count = get_title_pattern_info(last_title_content)
                        if last_priority >= 0:
                            last_pattern_key = (last_priority, last_dot_count)
                            if last_pattern_key in pattern_to_level:
                                last_title_level_in_path = pattern_to_level[last_pattern_key]
                    
                    # 判断同级标题：只要 level 相同，就是同级标题，需要替换最后一个
                    # 使用和"章节层级详情"完全一样的逻辑：如果两个标题的 level 相同（都来自 adjust_title_levels），那么它们是同级的
                    if (len(current_section_path) > 0 and 
                        last_title_level_in_path is not None and
                        last_title_level_in_path == level):
                        # 同级标题（level 相同），替换最后一个
                        current_section_path[-1] = block_content
                    else:
                        # 子级标题或更高级的标题，追加到路径
                        # 如果 path_depth == len(current_section_path)，说明是子级标题，追加
                        # 如果 path_depth < len(current_section_path)，说明是更高级的标题，已经通过上面的 while 循环截断了，现在追加
                        if len(current_section_path) == path_depth:
                            current_section_path.append(block_content)
                        else:
                            # 这种情况理论上不应该发生，但为了安全起见，还是追加
                            current_section_path.append(block_content)
                    
                    # 创建新章节
                    # 标准化 section_path 用于存储（去掉前导和尾随空格）
                    normalized_section_path = [title.strip() for title in current_section_path]
                    section_path_str = " > ".join(normalized_section_path)
                    
                    # 使用xxhash方式生成section_id（section_path + doc_id），与段落chunks的parent_section_id生成方式一致
                    import xxhash
                    if section_path_str and doc_id:
                        section_id = xxhash.xxh64((section_path_str + doc_id).encode("utf-8", "surrogatepass")).hexdigest()
                    elif section_path_str:
                        # 如果没有doc_id，使用section_path作为临时值（理论上不应该发生）
                        section_id = section_path_str
                    else:
                        section_id = ""
                    
                    # 如果存在上一个章节，先保存它（不聚合内容，内容聚合在 _apply_paper_merge_strategy 中完成）
                    if current_section:
                        current_section["paragraph_chunk_ids"] = current_section_paragraph_ids.copy()
                        sections.append(current_section)
                    
                    # 创建新章节
                    # 获取当前标题的行号（直接从 block 计算，不依赖 title_content_to_line）
                    # 每个 block 都有 page_index（在 _call_ocr_api 中已添加，或从 pagexxx 文件读取时已添加）
                    block_page_index = block.get("page_index", 1)
                    block_order = block.get("block_order")
                    block_id = block.get("block_id", 0)
                    if block_order is not None:
                        last_title_line = block_page_index * 10000 + block_order
                    elif block_id > 0:
                        last_title_line = block_page_index * 10000 + block_id
                    else:
                        last_title_line = block_page_index * 10000 + 1
                    
                    current_section = {
                        "section_id": section_id,  # 使用xxhash方式生成的id（section_path + doc_id），与段落chunks的parent_section_id一致
                        "section_path": normalized_section_path.copy(),  # 使用归一化后的section_path
                        "level": level,
                        "page_index": page_index,  # 保存标题出现的页码（用于与 gen_title_report.py 保持一致）
                        "line": last_title_line,  # 保存最后一个标题的行号（用于与 gen_title_report.py 保持一致）
                        "paragraph_chunk_ids": [],
                        "content": ""  # 内容会在后续段落中填充
                    }
                    current_section_paragraph_ids = []
                    
                    logger.debug(f"[段落提取] 发现标题: '{block_content}' (level={level}, section_id={section_id})")
            
            # 处理段落（text）和摘要（abstract）
            elif block_label == "text" or block_label == "abstract":
                # 对于段落内容，去掉所有前后空格
                block_content = block.get("block_content", "").strip()
                
                # 跳过空内容
                if not block_content:
                    continue
                
                # 判断是否为短段落
                is_short = self._is_short_paragraph(block_content)
                
                # ========== 修复：检查中间是否有标题 ==========
                # 检查是否需要跨页拼接（只有跨页且中间没有 paragraph_title/doc_title 时才合并）
                # 如果just_encountered_title为True，说明中间有标题，不应该跨页合并
                if last_text_block and last_page_index and page_index == last_page_index + 1 and not just_encountered_title:
                # ========== 修复结束 ==========
                    # 跨页拼接：将上一个页面的最后一个 text 和当前页面的第一个 text 合并
                    # 这是因为 PaddleOCR 错误地将一个段落识别为两个段落
                    combined_content = last_text_block.get("block_content", "").strip() + block_content
                    
                    # 如果last_text_block是短段落，并且已经在pending_short_paragraphs中，需要移除它
                    # 因为跨页合并已经处理了这个短段落
                    if pending_short_paragraphs:
                        last_block_content = last_text_block.get("block_content", "").strip()
                        # 检查pending_short_paragraphs中是否有与last_text_block相同的内容
                        pending_short_paragraphs = [
                            p for p in pending_short_paragraphs 
                            if p.get("block_content", "").strip() != last_block_content
                        ]
                        # 如果pending_short_paragraphs被清空，重置pending_short_paragraphs_page_index
                        if not pending_short_paragraphs:
                            pending_short_paragraphs_page_index = None
                    
                    # 如果合并后的内容仍然是短段落，且当前block也是短段落，加入待合并列表
                    if self.enable_short_paragraph_merge and is_short and self._is_short_paragraph(combined_content):
                        # 清空last_text_block，将合并后的内容加入待合并列表
                        if pending_short_paragraphs_page_index is None:
                            pending_short_paragraphs_page_index = last_page_index
                        pending_short_paragraphs.append({
                            "block_content": combined_content,
                            "page_index": last_page_index
                        })
                        last_text_block = None
                        last_page_index = None
                    else:
                        # 合并后不是短段落，或者短段落合并未启用，直接创建段落
                        paragraph_counter += 1
                        para_chunk_id = f"{filename}_para_{paragraph_counter}"
                        paragraph = {
                            "content": combined_content,
                            "page_index": last_page_index,  # 使用第一个段落的页码
                            "section_path": current_section_path.copy(),
                            "parent_section_id": get_parent_section_id(para_chunk_id),
                            "chunk_id": para_chunk_id
                        }
                        paragraphs.append(paragraph)
                        if current_section:
                            current_section_paragraph_ids.append(para_chunk_id)
                        
                        last_text_block = None
                        last_page_index = None
                elif is_short and self.enable_short_paragraph_merge:
                    # 当前block是短段落，先处理上一个非短段落（如果有）
                    # 注意：
                    # 1. 如果last_text_block是短段落，并且已经在pending_short_paragraphs中，不应该重复处理
                    # 2. 如果just_encountered_title为True，说明last_text_block已经在处理paragraph_title时被处理了，不应该重复处理
                    if last_text_block and not just_encountered_title:
                        last_block_content = last_text_block.get("block_content", "").strip()
                        # 检查last_text_block是否已经在pending_short_paragraphs中
                        is_in_pending = any(
                            p.get("block_content", "").strip() == last_block_content 
                            for p in pending_short_paragraphs
                        )
                        # 只有当last_text_block不在pending_short_paragraphs中时，才处理它
                        if not is_in_pending:
                            paragraph_counter += 1
                            para_chunk_id = f"{filename}_para_{paragraph_counter}"
                            paragraph = {
                                "content": last_block_content,
                                "page_index": last_page_index or 1,
                                "section_path": current_section_path.copy(),
                                "parent_section_id": get_parent_section_id(para_chunk_id),
                                "chunk_id": para_chunk_id
                            }
                            paragraphs.append(paragraph)
                            if current_section:
                                current_section_paragraph_ids.append(para_chunk_id)
                        last_text_block = None
                        last_page_index = None
                    
                    # 将当前短段落加入待合并列表
                    if pending_short_paragraphs_page_index is None:
                        pending_short_paragraphs_page_index = page_index
                    pending_short_paragraphs.append({
                        "block_content": block_content,
                        "page_index": page_index
                    })
                    
                    # 保存当前 text block，等待下一个 block 判断是否需要跨页拼接
                    # 即使是短段落，也需要保存，以便与下一个 text block 进行跨页合并
                    block_page_index = block.get("page_index", page_index)
                    last_text_block = block
                    last_page_index = block_page_index
                else:
                    # 当前block不是短段落，先处理待合并的短段落（如果有）
                    # 在处理pending_short_paragraphs之前，先检查last_text_block是否在其中
                    last_block_in_pending = False
                    if last_text_block and pending_short_paragraphs:
                        last_block_content = last_text_block.get("block_content", "").strip()
                        last_block_in_pending = any(
                            p.get("block_content", "").strip() == last_block_content 
                            for p in pending_short_paragraphs
                        )
                    
                    if pending_short_paragraphs:
                        # 使用分批合并策略处理短段落
                        paragraph_counter = self._merge_short_paragraphs_in_batches(
                            pending_short_paragraphs,
                            pending_short_paragraphs_page_index,
                            current_section_path,
                            get_parent_section_id,
                            filename,
                            paragraph_counter,
                            paragraphs,
                            current_section,
                            current_section_paragraph_ids
                        )
                        
                        pending_short_paragraphs = []
                        pending_short_paragraphs_page_index = None
                    
                    # ========== 修复：如果just_encountered_title为True，说明last_text_block已经在处理paragraph_title时被处理了 ==========
                    # 处理上一个非短段落（如果有）
                    # 注意：
                    # 1. 如果last_text_block已经在pending_short_paragraphs中被处理了，不应该重复处理
                    # 2. 如果just_encountered_title为True，说明last_text_block已经在处理paragraph_title时被处理了，不应该重复处理
                    if last_text_block and not last_block_in_pending and not just_encountered_title:
                    # ========== 修复结束 ==========
                        paragraph_counter += 1
                        para_chunk_id = f"{filename}_para_{paragraph_counter}"
                        paragraph = {
                            "content": last_text_block.get("block_content", "").strip(),
                            "page_index": last_page_index or 1,
                            "section_path": current_section_path.copy(),
                            "parent_section_id": get_parent_section_id(para_chunk_id),
                            "chunk_id": para_chunk_id
                        }
                        paragraphs.append(paragraph)
                        if current_section:
                            current_section_paragraph_ids.append(para_chunk_id)
                    
                    # 保存当前 text block，等待下一个 block 判断是否需要跨页拼接
                    last_text_block = block
                    last_page_index = page_index
                    
                    # ========== 修复：重置标志 ==========
                    # 已经处理了text block，不再是"刚刚遇到标题"的状态
                    just_encountered_title = False
                    # ========== 修复结束 ==========
            
            # 忽略其他类型的 block（figure_title, table 等）
            else:
                # ========== 修复：移除else分支中的paragraph_title处理逻辑，避免重复处理 ==========
                # paragraph_title已经在if分支中被处理了，不会进入else分支
                # 如果paragraph_title被continue跳过了（比如不是有效标题），也不应该在这里处理last_text_block
                # 因为无效的paragraph_title不应该影响段落的处理
                # 其他类型的 block（footnote、number、header 等）不影响跨页拼接，直接跳过
                # ========== 修复结束 ==========
                
                # ========== 修复：重置标志 ==========
                # 其他类型的block（除了paragraph_title和doc_title）也会重置标志
                # 这样确保只有紧跟在标题后面的text block才会检查just_encountered_title
                just_encountered_title = False
                # ========== 修复结束 ==========
        
        # 处理最后一个未完成的段落
        # 先处理待合并的短段落（如果有）
        # 在处理pending_short_paragraphs之前，先检查last_text_block是否在其中
        last_block_in_pending = False
        if last_text_block and pending_short_paragraphs:
            last_block_content = last_text_block.get("block_content", "").strip()
            last_block_in_pending = any(
                p.get("block_content", "").strip() == last_block_content 
                for p in pending_short_paragraphs
            )
        
        if pending_short_paragraphs:
            # 使用分批合并策略处理短段落
            paragraph_counter = self._merge_short_paragraphs_in_batches(
                pending_short_paragraphs,
                pending_short_paragraphs_page_index,
                current_section_path,
                get_parent_section_id,
                filename,
                paragraph_counter,
                paragraphs,
                current_section,
                current_section_paragraph_ids
            )
        
        # 处理最后一个非短段落（如果有）
        # 注意：如果last_text_block已经在pending_short_paragraphs中被处理了，不应该重复处理
        if last_text_block and not last_block_in_pending:
            paragraph_counter += 1
            para_chunk_id = f"{filename}_para_{paragraph_counter}"
            paragraph = {
                "content": last_text_block.get("block_content", "").strip(),
                "page_index": last_page_index or 1,
                "section_path": current_section_path.copy(),
                "parent_section_id": get_parent_section_id(para_chunk_id),
                "chunk_id": para_chunk_id
            }
            paragraphs.append(paragraph)
            if current_section:
                current_section_paragraph_ids.append(para_chunk_id)
        
        # 保存最后一个章节（不聚合内容，内容聚合在 _apply_paper_merge_strategy 中完成）
        if current_section:
            current_section["paragraph_chunk_ids"] = current_section_paragraph_ids.copy()
            sections.append(current_section)
        
        return paragraphs, sections
    
    def _create_paragraph_chunk(self, paragraph: Dict[str, Any], filename: str, base_doc: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        创建段落级 chunk（用于检索）
        
        Args:
            paragraph: 段落信息字典
            filename: 文件名
            base_doc: 基础文档结构（包含 docnm_kwd 等字段）
            **kwargs: 其他参数（可能包含 doc_id, kb_id 等）
        
        Returns:
            chunk 字典，如果创建失败返回 None
        """
        try:
            content = paragraph.get("content", "").strip()
            if not content:
                return None
            
            section_path = paragraph.get("section_path", [])
            parent_section_id = paragraph.get("parent_section_id", "")
            if parent_section_id is None:
                parent_section_id = ""  # 将None转换为空字符串
            page_index = paragraph.get("page_index", 1)
            chunk_id = paragraph.get("chunk_id", "")
            
            # 构建章节路径字符串（标准化：去掉前导和尾随空格，确保与 section_id 生成时一致）
            normalized_section_path = [title.strip() for title in section_path] if section_path else []
            section_path_str = " > ".join(normalized_section_path) if normalized_section_path else ""
            
            # 调试：如果section_path存在但parent_section_id为空，记录警告
            if section_path_str and not parent_section_id:
                logger.warning(f"[创建段落Chunk] 警告: section_path存在但parent_section_id为空! "
                             f"section_path={section_path_str}, chunk_id={chunk_id}, "
                             f"paragraph_parent_section_id={paragraph.get('parent_section_id')}")
            
            # 构建章节标题前缀（保持与原有格式兼容：[章节标题]\n段落内容）
            section_title_prefix = ""
            if section_path_str:
                section_title_prefix = "[" + section_path_str + "]\n"
            
            # 构建 content_with_weight（格式：[章节标题]\n段落内容，与原有格式兼容）
            content_with_weight = section_title_prefix + content
            
            # 创建 chunk
            chunk = {
                "id": chunk_id,
                "content_with_weight": content_with_weight,
                "page_num_int": [page_index],  # 列表格式
                "parent_section_id": parent_section_id or "",
                "section_path": section_path_str,  # 章节路径字符串
                "chunk_type": "paragraph",  # 标记为段落级 chunk
                "docnm_kwd": base_doc.get("docnm_kwd", ""),
                "title_tks": base_doc.get("title_tks", ""),
                "title_sm_tks": base_doc.get("title_sm_tks", ""),
                "doc_id": kwargs.get("doc_id", ""),
                "kb_id": kwargs.get("kb_id", ""),
                "position_int": [[page_index, 0, 0, 0, 0]],  # 简化位置信息
                "top_int": [0],
                "doc_type_kwd": "text"  # 标记为文本类型
            }
            
            # 调用tokenize生成content_ltks等字段（用于检索）
            tokenize(chunk, content, False)  # 假设是中文文档
            
            # 为包含章节路径的文本内容添加重要关键词字段（与旧代码保持一致，section_title会触发大模型提取关键词）
            if section_path_str:
                content_for_keywords = section_path_str
                chunk["important_kwd"], chunk["important_tks"] = self._generate_keywords(
                    content_for_keywords, topn=8, context=f"paragraph_chunk section_path '{section_path_str}'"
                )
            
            return chunk
            
        except Exception as e:
            logger.error(f"[创建段落Chunk] 创建段落chunk失败: {str(e)}", exc_info=True)
            return None
    
    def _create_section_chunk(self, section: Dict[str, Any], filename: str, base_doc: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        创建章节级 chunk（用于返回）
        
        Args:
            section: 章节信息字典
            filename: 文件名
            base_doc: 基础文档结构（包含 docnm_kwd 等字段）
            **kwargs: 其他参数（可能包含 doc_id, kb_id 等）
        
        Returns:
            chunk 字典，如果创建失败返回 None
        """
        try:
            section_id = section.get("section_id", "")
            section_path = section.get("section_path", [])
            content = section.get("content", "").strip()  # 可能为空，将在 _apply_paper_merge_strategy 中填充
            paragraph_chunk_ids = section.get("paragraph_chunk_ids", [])
            level = section.get("level", 2)  # 获取标题层级（从 adjust_title_levels 分配）
            page_index = section.get("page_index", 1)  # 获取标题出现的页码（与 gen_title_report.py 保持一致）
            line = section.get("line", 0)  # 获取最后一个标题的行号（与 gen_title_report.py 保持一致）
            
            if not section_id:
                return None
            
            # 构建章节路径字符串
            section_path_str = " > ".join(section_path) if section_path else ""
            
            # 构建 content_with_weight（格式：[章节标题]\n章节内容，与原有格式兼容）
            if section_path_str:
                content_with_weight = "[" + section_path_str + "]\n" + content
            else:
                content_with_weight = content
            
            # 创建 chunk（章节级 chunk 的 ID 使用 section_path + doc_id 的 xxhash，与段落chunks的parent_section_id一致）
            # section_id已经在上面生成（使用xxhash方式）
            chunk = {
                "id": section_id,  # 使用xxhash方式生成的id（section_path + doc_id）
                "content_with_weight": content_with_weight,
                "parent_section_id": "",  # 章节级 chunk 没有父章节
                "section_path": section_path_str,  # 章节路径字符串
                "paragraph_chunk_ids": paragraph_chunk_ids,  # 关联的段落 chunk IDs
                "chunk_type": "section",  # 标记为章节级 chunk
                "level": level,  # 标题层级（从 adjust_title_levels 分配，与 gen_title_report.py 保持一致）
                "line": line,  # 最后一个标题的行号（与 gen_title_report.py 保持一致）
                "docnm_kwd": base_doc.get("docnm_kwd", ""),
                "title_tks": base_doc.get("title_tks", ""),
                "title_sm_tks": base_doc.get("title_sm_tks", ""),
                "doc_id": kwargs.get("doc_id", ""),
                "kb_id": kwargs.get("kb_id", ""),
                "page_num_int": [page_index],  # 使用标题出现的页码（与 gen_title_report.py 保持一致）
                "position_int": [],
                "top_int": [],
                "doc_type_kwd": "text"  # 标记为文本类型
            }
            
            return chunk
            
        except Exception as e:
            logger.error(f"[创建章节Chunk] 创建章节chunk失败: {str(e)}", exc_info=True)
            return None
    
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
                content_for_keywords, topn=8, context=f"text_content '{section_title}'"
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
                content_for_keywords, topn=8, context=f"text_content '{section_title}'"
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
        
        注意：对于段落级（chunk_type="paragraph"）和章节级（chunk_type="section"）chunks，
        不进行合并，因为它们已经按照新的设计进行了处理。
        """
        if not chunks:
            logger.info(f"[合并策略] 没有chunks需要合并: {filename}")
            return chunks
        
        logger.info(f"[合并策略] 开始应用合并策略: {filename}, 原始chunks数: {len(chunks)}")
        
        # 检查是否有段落级或章节级 chunks（新设计）
        has_new_chunk_types = any(chunk.get('chunk_type') in ('paragraph', 'section') for chunk in chunks)
        
        if has_new_chunk_types:
            # 处理新设计的 chunks（段落级和章节级）
            return self._apply_new_chunk_merge_strategy(chunks, filename)
        
        # 按section_title分组（原有逻辑，用于其他类型的chunks，如 textin 的旧格式）
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
    
    def _apply_new_chunk_merge_strategy(self, chunks: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """
        处理新设计的 chunks（段落级和章节级）的合并策略
        
        1. 对于 section 类型的 chunks：根据 paragraph_chunk_ids 聚合内容
        2. 对于没有 section_path 的段落：聚合到一个"无标题章节"中
        """
        paragraph_chunks = [c for c in chunks if c.get('chunk_type') == 'paragraph']
        section_chunks = [c for c in chunks if c.get('chunk_type') == 'section']
        other_chunks = [c for c in chunks if c.get('chunk_type') not in ('paragraph', 'section')]
        
        # 创建段落 ID 到段落 chunk 的映射
        paragraph_map = {chunk.get('id', ''): chunk for chunk in paragraph_chunks}
        
        # 处理 section chunks：聚合内容
        processed_section_chunks = []
        for section_chunk in section_chunks:
            original_id = section_chunk.get('id', '')
            original_section_path = section_chunk.get('section_path', '')
            section_chunk = copy.deepcopy(section_chunk)
            new_id = section_chunk.get('id', '')
            if original_id != new_id:
                logger.warning(f"[合并策略] section_chunk id被修改! 原始id={original_id}, 新id={new_id}, section_path={original_section_path}")
            logger.debug(f"[合并策略] 处理section_chunk: id={new_id}, section_path={original_section_path[:60]}")
            paragraph_chunk_ids = section_chunk.get('paragraph_chunk_ids', [])
            
            # 如果 paragraph_chunk_ids 是字符串，尝试解析为 JSON
            if isinstance(paragraph_chunk_ids, str):
                try:
                    import json
                    paragraph_chunk_ids = json.loads(paragraph_chunk_ids)
                except:
                    paragraph_chunk_ids = []
            
            # 聚合段落内容
            section_content_parts = []
            for para_chunk_id in paragraph_chunk_ids:
                para_chunk = paragraph_map.get(para_chunk_id)
                if para_chunk:
                    # 从段落 chunk 的 content_with_weight 中提取内容
                    content_with_weight = para_chunk.get('content_with_weight', '')
                    if content_with_weight.startswith('[') and ']\n' in content_with_weight:
                        # 去掉章节标题前缀，只保留段落内容
                        content = content_with_weight.split(']\n', 1)[1]
                    else:
                        content = content_with_weight
                    section_content_parts.append(content)
            
            # 更新 section chunk 的内容
            section_content = "\n\n".join(section_content_parts)
            section_path_str = section_chunk.get('section_path', '')
            
            if section_path_str:
                section_chunk['content_with_weight'] = f"[{section_path_str}]\n{section_content}"
            else:
                section_chunk['content_with_weight'] = section_content
            
            processed_section_chunks.append(section_chunk)
        
        # 处理没有 section_path 的段落：聚合到一个"无标题章节"中
        no_title_paragraphs = [c for c in paragraph_chunks if not c.get('section_path', '').strip()]
        
        if no_title_paragraphs:
            # 创建无标题章节
            # 使用xxhash方式生成no_title_section_id（__no_title__ + doc_id），与段落chunks的parent_section_id生成方式一致
            import xxhash
            doc_id = no_title_paragraphs[0].get('doc_id', '') if no_title_paragraphs else ''
            if doc_id:
                no_title_section_id = xxhash.xxh64((f"__no_title___{doc_id}").encode("utf-8", "surrogatepass")).hexdigest()
            else:
                # 如果没有 doc_id，使用原来的方式（理论上不应该发生）
                no_title_section_id = xxhash.xxh64("__no_title__".encode("utf-8", "surrogatepass")).hexdigest()
            no_title_paragraph_ids = [c.get('id', '') for c in no_title_paragraphs]
            
            # 聚合内容，并更新无标题段落的parent_section_id
            no_title_content_parts = []
            for para_chunk in no_title_paragraphs:
                # 更新parent_section_id指向无标题章节
                para_chunk['parent_section_id'] = no_title_section_id
                content_with_weight = para_chunk.get('content_with_weight', '')
                if content_with_weight.startswith('[') and ']\n' in content_with_weight:
                    content = content_with_weight.split(']\n', 1)[1]
                else:
                    content = content_with_weight
                no_title_content_parts.append(content)
            
            no_title_content = "\n\n".join(no_title_content_parts)
            
            # 创建无标题章节 chunk
            no_title_section_chunk = {
                "id": no_title_section_id,
                "content_with_weight": no_title_content,
                "parent_section_id": "",
                "section_path": "",
                "paragraph_chunk_ids": no_title_paragraph_ids,
                "chunk_type": "section",
                "docnm_kwd": no_title_paragraphs[0].get('docnm_kwd', '') if no_title_paragraphs else '',
                "title_tks": no_title_paragraphs[0].get('title_tks', '') if no_title_paragraphs else '',
                "title_sm_tks": no_title_paragraphs[0].get('title_sm_tks', '') if no_title_paragraphs else '',
                "doc_id": no_title_paragraphs[0].get('doc_id', '') if no_title_paragraphs else '',
                "kb_id": no_title_paragraphs[0].get('kb_id', '') if no_title_paragraphs else '',
                "page_num_int": [],
                "position_int": [],
                "top_int": [],
                "doc_type_kwd": "text"
            }
            processed_section_chunks.append(no_title_section_chunk)
        
        # 返回：所有段落 chunks + 处理后的章节 chunks + 其他 chunks
        # 注意：段落级 chunks 全部保留（用于检索），无 section_path 的段落也会被聚合到无标题章节中（用于返回）
        result_chunks = paragraph_chunks + processed_section_chunks + other_chunks
        
        logger.info(f"[合并策略] 新设计chunks处理完成: {filename}, {len(chunks)} -> {len(result_chunks)} chunks "
                   f"(段落: {len(paragraph_chunks)}, 章节: {len(processed_section_chunks)}, 其他: {len(other_chunks)})")
        
        return result_chunks
    
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
        # 如果binary为None，尝试从存储读取
        if binary is None:
            logger.warning(f"[解析入口] binary参数为None，尝试从存储读取文件: {filename}")
            # 从kwargs中获取kb_id，用于从存储读取文件
            kb_id = kwargs.get("kb_id")
            if kb_id:
                try:
                    from rag.utils.storage_factory import STORAGE_IMPL
                    binary = STORAGE_IMPL.get(kb_id, filename)
                    if binary:
                        logger.info(f"[解析入口] 从存储读取文件成功: {filename}, 大小: {len(binary)} bytes")
                    else:
                        logger.error(f"[解析入口] 从存储读取文件失败: {filename} (返回None)")
                        return []
                except Exception as e:
                    logger.exception(f"[解析入口] 从存储读取文件异常: {filename}, 错误: {e}")
                    return []
            else:
                logger.error(f"[解析入口] binary为None且无法从存储读取（缺少kb_id）: {filename}")
                return []
        
        # 获取解析配置
        parser_config = kwargs.get("parser_config", {})
        custom_config = parser_config.get("custom_config", {})
        
        # 添加tenant_id到custom_config中，用于LLM模型初始化
        if "tenant_id" in kwargs:
            custom_config["tenant_id"] = kwargs["tenant_id"]
        
        # bucket配置会由 CustomPdfParser._init_bucket_config() 自动处理
        # 优先级：环境变量 > custom_config > 配置文件 > 默认值
        
        logger.info(f"[解析器初始化] Parser config: {parser_config}")
        
        # 创建自定义解析器实例（传递parser_config用于短段落合并配置）
        parser = CustomPdfParser(custom_config=custom_config, parser_config=parser_config)
        
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
        
        # section_title 字段已统一使用 section_path，无需删除
        
        # 过滤掉标题类型的chunks，只返回文本内容
        filtered_chunks = [c for c in processed_chunks if c.get("doc_type_kwd") != "title"]
        
        # 移除不在 schema 中的字段（用于内部逻辑，不需要插入数据库）
        # level: 保留用于 test_custom_chunk.py 等工具脚本，在 infinity_conn.py 的 insert 方法中移除
        # line: 保留用于 test_custom_chunk.py 等工具脚本，在 infinity_conn.py 的 insert 方法中移除
        # chunk_type: 保留用于 task_executor.py 区分段落级和章节级，在 infinity_conn.py 的 insert 方法中移除
        # 不再在这里移除 level 和 line，改为在 infinity_conn.py 中统一处理
        
        return filtered_chunks
        
    except Exception as e:
        logger.error(f"Custom parser error for {filename}: {str(e)}")
        if callback:
            callback(-1, f"Custom parser error: {str(e)}")
        raise