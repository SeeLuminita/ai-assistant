"""
LangChain Day 3: Agent 与 Tools
- 什么是 Agent
- 工具调用
- 让 LLM 自己决定用什么工具
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ============================================
# 1. 定义工具（Tools）
# ============================================
print("=== 1. 定义工具 ===")

@tool
def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间是: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，气温 25°C",
        "上海": "多云，气温 28°C",
        "深圳": "小雨，气温 30°C",
        "成都": "阴天，气温 22°C"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如: calculate("2+3*4")"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

print(f"已定义工具: {[t.name for t in [get_current_time, get_weather, calculate]]}")
print()

# ============================================
# 2. 创建 Agent
# ============================================
print("=== 2. 创建 Agent ===")

tools = [get_current_time, get_weather, calculate]

# 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，可以使用工具来帮助用户。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("Agent 创建完成")
print()

# ============================================
# 3. 测试 Agent
# ============================================
print("=== 3. 测试 Agent ===")

# 测试 1: 问时间（Agent 应该调用 get_current_time 工具）
print("问题 1: 现在几点了？")
result1 = agent_executor.invoke({"input": "现在几点了？"})
print(f"回答: {result1['output']}")
print()

# 测试 2: 问天气（Agent 应该调用 get_weather 工具）
print("问题 2: 北京今天天气怎么样？")
result2 = agent_executor.invoke({"input": "北京今天天气怎么样？"})
print(f"回答: {result2['output']}")
print()

# 测试 3: 计算（Agent 应该调用 calculate 工具）
print("问题 3: 帮我算一下 123 * 456")
result3 = agent_executor.invoke({"input": "帮我算一下 123 * 456"})
print(f"回答: {result3['output']}")
print()

# 测试 4: 复杂问题（Agent 自己决定调用哪些工具）
print("问题 4: 现在几点了？北京天气怎么样？顺便帮我算一下 100 + 200")
result4 = agent_executor.invoke({"input": "现在几点了？北京天气怎么样？顺便帮我算一下 100 + 200"})
print(f"回答: {result4['output']}")
