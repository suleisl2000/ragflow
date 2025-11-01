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
PICO检索工具函数
提供PICO检索策略判断和相关辅助函数
"""

import logging

# PICO检索支持（可选导入，如果模块不存在不影响主功能）
try:
    from rag.retrieval import PICORetriever, UMLSMapper
    PICO_RETRIEVAL_AVAILABLE = True
except ImportError:
    PICO_RETRIEVAL_AVAILABLE = False
    PICORetriever = None
    UMLSMapper = None


def _is_evidence_based_medical_query(question: str, chat_mdl) -> bool:
    """判断是否为循证医学查询（需要使用PICO框架）"""
    if not question:
        return False
    try:
        prompt = f"""请判断以下医疗查询是否为循证医学查询（需要PICO框架的结构化检索）。
        循证医学查询通常包含：
        - P (Population/Patient): 人群/患者特征
        - I (Intervention): 干预措施（治疗、药物等）
        - O (Outcome): 结局指标（效果、疗效等）
        - 可能包含：对比（C: Comparison）
        查询：{question}
        请只回答"是"或"否"。
        回答："""
        # LLMBundle.chat() 需要 system 和 history 作为位置参数
        response = chat_mdl.chat(
            system="",
            history=[{"role": "user", "content": prompt}],
            gen_conf={}
        )
        result = response.strip().lower()
        is_evidence_based = "是" in result or "yes" in result or result.startswith("y")
        logging.info(f"[PICO判断] 查询: {question[:100]}..., 判断结果: {is_evidence_based}")
        return is_evidence_based
    except Exception as e:
        logging.warning(f"[PICO判断] LLM判断失败，使用关键词回退: {e}")
        evidence_keywords = ["能否", "是否", "效果", "疗效", "治疗", "干预", "患者", "人群", "结局", "结果", "对比", "比较", "预防", "改善", "降低", "提高", "减少", "增加"]
        count = sum(1 for kw in evidence_keywords if kw in question)
        return count >= 2


def _get_retrieval_strategy(req: dict, question: str, chat_mdl=None) -> str:
    """获取检索策略：pico, standard, auto"""
    strategy = req.get("retrieval_strategy", "").lower()
    if not strategy:
        use_pico = req.get("use_pico", False)
        if use_pico:
            strategy = "pico"
        else:
            strategy = "standard"
    
    if strategy == "pico":
        logging.info(f"[检索策略] 使用PICO检索（显式指定）")
        return "pico"
    elif strategy == "standard":
        logging.info(f"[检索策略] 使用标准检索（显式指定）")
        return "standard"
    elif strategy == "auto":
        if not PICO_RETRIEVAL_AVAILABLE:
            logging.info(f"[检索策略] PICO模块不可用，使用标准检索")
            return "standard"
        if chat_mdl is None:
            logging.warning(f"[检索策略] auto模式需要chat_mdl，回退到标准检索")
            return "standard"
        is_evidence_based = _is_evidence_based_medical_query(question, chat_mdl)
        if is_evidence_based:
            logging.info(f"[检索策略] 自动判断为循证医学查询，使用PICO检索")
            return "pico"
        else:
            logging.info(f"[检索策略] 自动判断为非循证医学查询，使用标准检索")
            return "standard"
    else:
        logging.info(f"[检索策略] 未指定策略，默认使用标准检索")
        return "standard"



