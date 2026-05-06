import os
from datetime import datetime
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

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    model = st.selectbox("模型", ["qwen-turbo", "qwen-plus", "qwen-max"])
    temperature = st.slider("创造性", 0.0, 1.0, 0.7)
    system_prompt = st.text_area(
        "角色设定", 
        value="你是一位专业的AI助手，回答问题要准确、有条理、有帮助。"
    )

# 标题
st.title("🤖 AI 智能助手")
st.write("输入问题，AI 为你解答")

# 显示当前时间
now = datetime.now()
st.caption(f"📅 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# 输入框
user_input = st.text_input("你的问题：", placeholder="例如：今天星期几？")

# 点击按钮
if st.button("发送"):
    if user_input:
        # 构建带时间上下文的 prompt
        context = f"""当前时间信息：
- 日期: {now.strftime('%Y年%m月%d日')}
- 星期: {['一','二','三','四','五','六','日'][now.weekday()]}
- 时间: {now.strftime('%H:%M:%S')}

用户问题: {user_input}

请根据以上时间信息回答用户问题。"""
        
        with st.spinner("AI 思考中..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=temperature
            )
            answer = response.choices[0].message.content
        
        st.success("回答：")
        st.write(answer)
    else:
        st.warning("请输入问题")