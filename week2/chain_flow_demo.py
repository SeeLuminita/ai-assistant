"""
验证 Chain 的数据流转
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}。"),
    ("user", "{question}")
])

# 单独看每一步
input_data = {"role": "程序员", "question": "用一句话什么是API？"}

# 第一步：prompt 处理
print("=== 第1步: prompt 处理 ===")
step1_output = prompt.invoke(input_data)
print(f"类型: {type(step1_output)}")
print(f"内容: {step1_output}")
print()

# 第二步：llm 处理
print("=== 第2步: llm 处理 ===")
step2_output = llm.invoke(step1_output)
print(f"类型: {type(step2_output)}")
print(f"内容: {step2_output}")
print()

# 第三步：parser 处理
print("=== 第3步: parser 处理 ===")
parser = StrOutputParser()
step3_output = parser.invoke(step2_output)
print(f"类型: {type(step3_output)}")
print(f"内容: {step3_output}")
print()

# 对比：Chain 一步完成
print("=== Chain 一步完成 ===")
chain = prompt | llm | StrOutputParser()
result = chain.invoke(input_data)
print(f"结果: {result}")
