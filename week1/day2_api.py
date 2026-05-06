import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 百炼平台的 OpenAI 兼容接口
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def chat(message):
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {e}"

if __name__ == "__main__":
    result = chat("用一句话介绍Python")
    print("结果:", result)