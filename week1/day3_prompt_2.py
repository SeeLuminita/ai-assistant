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

# ========== 技巧 2：Few-shot（少样本示例）==========
print("=== 技巧 2：Few-shot 示例 ===")

# 没有 Few-shot
result1 = chat("把以下句子翻译成英文：苹果很好吃")
print(f"无示例: {result1}\n")

# 有 Few-shot - 给出示例
result2 = chat("""请按照以下格式翻译：

示例：
中文：今天天气很好
英文：The weather is nice today.

中文：我喜欢编程
英文：I like programming.

现在请翻译：
中文：苹果很好吃
英文：""")
print(f"有示例: {result2}\n")

# ========== 技巧 3：结构化输出 ==========
print("=== 技巧 3：结构化输出 ===")

result3 = chat("""分析以下代码的时间复杂度，请按以下格式输出：

代码：
for i in range(n):
    for j in range(n):
        print(i, j)

请输出：
- 时间复杂度：
- 空间复杂度：
- 解释：""")

print(result3)