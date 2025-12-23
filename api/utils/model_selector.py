#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型选择工具：根据使用场景选择正确的模型配置
"""

from api.utils.configs import get_base_config


def get_chat_model_for_scenario(tenant_id, scenario="query", llm_name=None, use_config_override=True):
    """
    根据使用场景获取Chat模型
    
    Args:
        tenant_id: 租户ID
        scenario: 使用场景
            - "indexing": 索引构建场景（关键词提取、问题生成等）
            - "query": 查询/对话场景
            - "other": 其他场景（默认使用查询场景的模型）
        llm_name: 如果指定，优先使用（用于Dialog中用户指定的模型）
        use_config_override: 如果为True，即使指定了llm_name，也优先使用配置文件中的模型（用于运行时覆盖数据库值）
    
    Returns:
        model_id: 模型ID字符串，格式为 "model_name@factory"，如果未配置则返回None
    """
    if scenario == "indexing":
        # 索引构建场景：使用 kb_default_llm.chat_model
        kb_llm_config = get_base_config("kb_default_llm", {}) or {}
        kb_default_models = kb_llm_config.get("default_models", {}) or {}
        chat_model = kb_default_models.get("chat_model", {}) or {}
        
        if chat_model.get("name"):
            model_name = chat_model.get("name")
            factory = chat_model.get("factory", "LocalAI")
            return f"{model_name}@{factory}"
    
    # 查询/对话场景或其他场景：使用 user_default_llm.chat_model
    user_llm_config = get_base_config("user_default_llm", {}) or {}
    user_default_models = user_llm_config.get("default_models", {}) or {}
    chat_model = user_default_models.get("chat_model", {}) or {}
    
    if chat_model.get("name"):
        model_name = chat_model.get("name")
        factory = chat_model.get("factory", "Tongyi-Qianwen")
        config_model_id = f"{model_name}@{factory}"
        
        # 如果配置了模型，优先使用配置的模型（覆盖数据库值）
        if use_config_override:
            return config_model_id
        # 如果未启用覆盖，且用户指定了模型，使用用户指定的模型
        elif llm_name:
            return llm_name
        else:
            return config_model_id
    
    # 如果配置文件中都没有，使用用户指定的模型或返回None
    if llm_name:
        return llm_name
    return None


def get_embedding_model_for_scenario(tenant_id, scenario="query", embd_id=None, use_config_override=True):
    """
    根据使用场景获取Embedding模型
    
    Args:
        tenant_id: 租户ID
        scenario: 使用场景
            - "indexing": 索引构建场景
            - "query": 查询/检索场景
            - "other": 其他场景（默认使用查询场景的模型）
        embd_id: 如果指定，优先使用（用于Knowledgebase中指定的模型）
        use_config_override: 如果为True，即使指定了embd_id，也优先使用配置文件中的模型（用于运行时覆盖数据库值）
    
    Returns:
        model_id: 模型ID字符串，格式为 "model_name@factory"，如果未配置则返回None
    """
    if scenario == "indexing":
        # 索引构建场景：使用 kb_default_llm.embedding_model
        kb_llm_config = get_base_config("kb_default_llm", {}) or {}
        kb_default_models = kb_llm_config.get("default_models", {}) or {}
        embedding_model = kb_default_models.get("embedding_model", {}) or {}
        
        if embedding_model.get("name"):
            model_name = embedding_model.get("name")
            factory = embedding_model.get("factory", "LocalAI")
            return f"{model_name}@{factory}"
    
    # 查询/检索场景或其他场景：使用 user_default_llm.embedding_model
    user_llm_config = get_base_config("user_default_llm", {}) or {}
    user_default_models = user_llm_config.get("default_models", {}) or {}
    embedding_model = user_default_models.get("embedding_model", {}) or {}
    
    if embedding_model.get("name"):
        model_name = embedding_model.get("name")
        factory = embedding_model.get("factory", "LocalAI")
        config_model_id = f"{model_name}@{factory}"
        
        # 如果配置了模型，优先使用配置的模型（覆盖数据库值）
        if use_config_override:
            return config_model_id
        # 如果未启用覆盖，且用户指定了模型，使用用户指定的模型
        elif embd_id:
            return embd_id
        else:
            return config_model_id
    
    # 如果配置文件中都没有，使用用户指定的模型或返回None
    if embd_id:
        return embd_id
    return None

