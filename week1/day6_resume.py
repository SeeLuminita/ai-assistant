import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

st.set_page_config(page_title="AI 简历润色器", page_icon="📄")

st.title("📄 AI 简历润色器")
st.write("上传简历，AI 帮你优化")

# 上传文件
uploaded_file = st.file_uploader("上传简历 (TXT 格式)", type=["txt"])

# 或直接输入
resume_text = st.text_area(
    "或直接粘贴简历内容",
    placeholder="姓名：张三\n工作经验：3年Java开发...",
    height=200
)

# 获取简历内容
content = None
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.info(f"已上传: {uploaded_file.name}")
elif resume_text:
    content = resume_text

# 润色选项
col1, col2,col3,col4 = st.columns(4)
with col1:
    target_job = st.text_input("目标岗位", placeholder="例如：Java开发工程师")
with col2:
    focus = st.selectbox("优化重点", ["整体润色", "项目经验", "技术能力", "自我评价"])
with col3:
    remark = st.text_input("意向行业",placeholder="例如：保险金融、互联网、传统软件")
with col4:
    remark = st.text_area("补充信息",placeholder="例如：本人当前已婚已育",height=100)
# 生成按钮
if st.button("🚀 开始润色", type="primary"):
    if content:
        with st.spinner("AI 分析中..."):
            prompt = f"""你是北大医疗资深的人力资源专家和简历优化顾问。请对以下简历进行专业润色。

简历内容：
{content}

目标岗位：{target_job if target_job else "通用"}
优化重点：{focus}

请提供：
1. 【简历评分】满分10分，给出评分和简短理由
2. 【问题分析】指出简历中的3-5个问题
3. 【优化建议】针对每个问题给出具体修改建议
4. 【润色后简历】优化后的完整简历内容

要求：
- 语言专业、简洁
- 突出亮点和成果
- 使用 STAR 法则描述项目经验（情境、任务、行动、结果）
- 量化工作成果（数字、百分比等）
"""

            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result = response.choices[0].message.content
        
        # 显示结果
        st.subheader("📊 分析结果")
        st.markdown(result)
        
        # 下载按钮
        st.download_button(
            label="📥 下载润色后的简历",
            data=result,
            file_name="优化简历.txt",
            mime="text/plain"
        )
    else:
        st.warning("请上传简历文件或粘贴简历内容")