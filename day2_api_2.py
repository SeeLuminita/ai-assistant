import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def chat(message):
    """普通调用"""
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

def chat_stream(message):
    """流式调用 - 逐字输出"""
    stream = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": message}],
        stream=True  # 开启流式
    )
    
    print("AI: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # 换行

if __name__ == "__main__":
    # 普通调用
    print("=== 普通调用 ===")
    result = chat("什么是Python？")
    print(f"AI: {result}")
    
    print("\n=== 流式调用 ===")
    chat_stream("什么是Python？请详细说明")