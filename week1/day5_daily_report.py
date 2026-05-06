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

st.set_page_config(page_title="AI 日报生成器", page_icon="📝")

st.title("📝 AI 日报生成器")
st.write("输入今天的工作内容，AI 帮你生成专业日报")

# 日期
today = datetime.now().strftime("%Y-%m-%d")
st.caption(f"📅 {today}")

# 输入区域
st.subheader("工作内容")
work_done = st.text_area(
    "今天完成了什么？",
    placeholder="例如：\n1. 完成了用户登录功能开发\n2. 修复了3个bug\n3. 参加了需求评审会议",
    height=150
)

# 选项
col1, col2 = st.columns(2)
with col1:
    style = st.selectbox("日报风格", ["简洁", "详细", "专业"])
with col2:
    language = st.selectbox("语言", ["中文", "英文"])

# 生成按钮
if st.button("🎯 生成日报", type="primary"):
    if work_done.strip():
        with st.spinner("生成中..."):
            prompt = f"""你是一位专业的职场助理。请根据以下工作内容生成一份{style}风格的日报。

今日工作内容：
{work_done}

要求：
1. 使用{language}
2. 结构清晰，分点列出
3. 语言专业、简洁
4. 格式如下：

【日报】{today}

一、今日工作
- ...

二、明日计划
- ...

三、遇到的问题
- （如果没有可写"无"）
"""

            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            report = response.choices[0].message.content
        
        # 显示结果
        st.subheader("📄 生成的日报")
        st.code(report, language=None)
        
        # 复制提示
        st.info("💡 选中上方日报内容，Ctrl+C 复制到钉钉/企业微信")
    else:
        st.warning("请输入工作内容")