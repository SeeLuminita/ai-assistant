import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def chat(prompt):
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========== 技巧 1：角色设定 ==========
print("=== 技巧 1：角色设定 ===")

# 差的 Prompt
result1 = chat("解释什么是递归")
print(f"普通回答:\n{result1}\n")

# 好的 Prompt - 设定角色
result2 = chat("""你是一位有10年经验的编程老师，擅长用通俗易懂的比喻解释复杂概念。
请用生活中的例子解释什么是递归。""")
print(f"角色设定后:\n{result2}\n")