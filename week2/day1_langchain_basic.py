"""
LangChain Day 1: 基础概念
- LLM 调用
- Prompt Template
- 简单 Chain
"""

import os
from dotenv import load_dotenv
from langchain_openai import SS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================
# 1. LLM 初始化
# ============================================
print("=== 1. LLM 初始化 ===")

llm = SS(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 简单调用
response = llm.invoke("用一句话介绍 LangChain")
print(f"回答: {response.content}")
print()

# ============================================
# 2. Prompt Template（提示词模板）
# ============================================
print("=== 2. Prompt Template ===")

# 创建模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，回答要{style}。"),
    ("user", "{question}")
])

# 填充模板
formatted_prompt = prompt.format(
    role="资深程序员",
    style="简洁专业",
    question="什么是 REST API？"
)

print(f"格式化后的 Prompt:\n{formatted_prompt}")
print()

# ============================================
# 3. Chain（链式调用）
# ============================================
print("=== 3. Chain 链式调用 ===")

# 创建链: prompt -> llm
chain = prompt | llm

# 调用链
response = chain.invoke({
    "role": "产品经理",
    "style": "通俗易懂",
    "question": "什么是微服务？"
})

print(f"回答: {response.content}")
print()

# ============================================
# 4. 输出解析器
# ============================================
print("=== 4. 输出解析器 ===")

from langchain_core.output_parsers import StrOutputParser

# 添加输出解析器到链
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "role": "技术面试官",
    "style": "专业详细",
    "question": "解释一下数据库索引"
})

print(f"解析后的输出:\n{response}")
