"""
LangChain Day 3: Agent 与 Tools（简化版）
- 什么是 Agent
- 工具调用
- 让 LLM 自己决定用什么工具
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间是: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    weather_data = {
        "北京": "晴天，气温 25°C",
        "上海": "多云，气温 28°C",
        "深圳": "小雨，气温 30°C",
        "成都": "阴天，气温 22°C"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

print("工具定义完成: get_current_time, get_weather, calculate")
print()

# ============================================
# 2. Agent 逻辑
# ============================================
print("=== 2. Agent 逻辑 ===")

# 工具注册表
tools = {
    "get_current_time": {"func": get_current_time, "desc": "获取当前时间，无需参数"},
    "get_weather": {"func": get_weather, "desc": "获取城市天气，需要参数: city"},
    "calculate": {"func": calculate, "desc": "计算数学表达式，需要参数: expression"}
}

def agent_chat(user_input: str) -> str:
    """Agent 对话函数"""
    
    # 构建工具说明
    tool_list = "\n".join([f"- {name}: {info['desc']}" for name, info in tools.items()])
    
    # 第一步：让 LLM 决定是否需要调用工具
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个AI助手，可以使用以下工具：
{tool_list}

当用户问题需要使用工具时，请只返回 JSON 格式（不要其他内容）：
格式: TOOL:工具名:参数值

例如：
- 查时间：TOOL:get_current_time:
- 查天气：TOOL:get_weather:北京
- 计算：TOOL:calculate:123*456

如果不需要工具，直接回答用户问题。"""),
        ("user", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"tool_list": tool_list, "input": user_input})
    
    # 检查是否是工具调用
    if response.startswith("TOOL:"):
        parts = response.strip().split(":")
        if len(parts) >= 2:
            tool_name = parts[1]
            tool_arg = parts[2] if len(parts) > 2 else ""
            
            if tool_name in tools:
                print(f"  [调用工具: {tool_name}]")
                
                # 调用工具
                func = tools[tool_name]["func"]
                if tool_arg:
                    result = func(tool_arg)
                else:
                    result = func()
                
                # 第二步：用工具结果回答用户
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是AI助手，请根据工具返回的结果回答用户问题。"),
                    ("user", "用户问题: {question}\n\n工具返回: {result}")
                ])
                final_chain = final_prompt | llm | StrOutputParser()
                return final_chain.invoke({"question": user_input, "result": result})
    
    # 不需要工具，直接返回
    return response

# ============================================
# 3. 测试 Agent
# ============================================
print("=== 3. 测试 Agent ===")

# 测试 1: 问时间
print("问题 1: 现在几点了？")
result1 = agent_chat("现在几点了？")
print(f"回答: {result1}")
print()

# 测试 2: 问天气
print("问题 2: 北京今天天气怎么样？")
result2 = agent_chat("北京今天天气怎么样？")
print(f"回答: {result2}")
print()

# 测试 3: 计算
print("问题 3: 帮我算一下 123 * 456")
result3 = agent_chat("帮我算一下 123 * 456")
print(f"回答: {result3}")
print()

# 测试 4: 普通问题
print("问题 4: Python 是什么？")
result4 = agent_chat("Python 是什么？")
print(f"回答: {result4[:100]}...")
