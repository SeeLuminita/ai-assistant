import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 页面配置
st.set_page_config(page_title="AI 助手", page_icon="🤖")

# 标题
st.title("🤖 AI 智能助手")
st.write("输入问题，AI 为你解答")

# 输入框
user_input = st.text_input("你的问题：", placeholder="例如：什么是 Python？")

# 点击按钮
if st.button("发送"):
    if user_input:
        with st.spinner("AI 思考中..."):
            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            answer = response.choices[0].message.content
        
        # 显示回答
        st.success("回答：")
        st.write(answer)
    else:
        st.warning("请输入问题")