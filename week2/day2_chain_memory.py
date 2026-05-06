"""
LangChain Day 2: Chain 与 Memory
- 多步 Chain
- 对话记忆
- LLM Chain
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ============================================
# 1. 简单链回顾
# ============================================
print("=== 1. 简单链 ===")

prompt = ChatPromptTemplate.from_template("告诉我一个关于{topic}的笑话")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "程序员"})
print(f"笑话: {result}")
print()

# ============================================
# 2. 多步链（Sequential Chain）
# ============================================
print("=== 2. 多步链 ===")

# 第一步：生成主题
prompt1 = ChatPromptTemplate.from_template("生成一个关于{industry}行业的创新点子")
chain1 = prompt1 | llm | StrOutputParser()

# 第二步：分析可行性
prompt2 = ChatPromptTemplate.from_template(
    "分析以下创业点子的可行性，给出3点建议：\n\n{idea}"
)
chain2 = prompt2 | llm | StrOutputParser()

# 组合链
print("正在生成创业点子...")
idea = chain1.invoke({"industry": "AI教育"})
print(f"点子: {idea[:100]}...\n")

print("正在分析可行性...")
analysis = chain2.invoke({"idea": idea})
print(f"分析: {analysis[:200]}...")
print()

# ============================================
# 3. RunnablePassthrough（数据传递）- 简化版
# ============================================
print("=== 3. 数据传递链 ===")

# 生成点子并显示
print("正在生成创业点子...")
result = chain1.invoke({"industry": "新能源汽车"})
print(f"结果:\n{result}")
print()

# ============================================
# 4. 对话记忆（Memory）
# ============================================
print("=== 4. 对话记忆 ===")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建聊天历史存储
chat_history = InMemoryChatMessageHistory()

# 创建带记忆的链
def get_session_history(session_id: str):
    return chat_history

# 系统提示词
system_prompt = "你是一个友好的AI助手，能够记住之前的对话内容。"
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])

# 创建带记忆的链
chain_with_history = prompt_with_history | llm | StrOutputParser()
chain_with_memory = RunnableWithMessageHistory(
    chain_with_history,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 第一次对话
print("第一次对话:")
response1 = chain_with_memory.invoke(
    {"input": "我叫张三，我是一名Java工程师"},
    config={"configurable": {"session_id": "test"}}
)
print(f"AI: {response1}")
print()

# 第二次对话（AI应该记得名字和职业）
print("第二次对话:")
response2 = chain_with_memory.invoke(
    {"input": "你还记得我叫什么名字吗？我的职业是什么？"},
    config={"configurable": {"session_id": "test"}}
)
print(f"AI: {response2}")
