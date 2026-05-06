"""
LangChain Day 5: RAG（真实 Embedding 版）
- 使用 HuggingFace 真实 Embedding 模型
- 文档向量化
- RAG 智能问答
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ============================================
# 1. 初始化真实 Embedding 模型
# ============================================
print("=== 1. 初始化 Embedding 模型 ===")
print("正在加载真实 Embedding 模型...")
print("首次运行需要下载模型（约 400MB），请耐心等待...")
print()

# 使用中文 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文向量模型，约 100MB
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("✓ Embedding 模型加载完成")
print()

# ============================================
# 2. 准备知识库文档
# ============================================
print("=== 2. 准备知识库文档 ===")

documents = [
    Document(
        page_content="""
公司请假制度：
1. 年假：员工每年享有5-15天年假，根据工龄确定
   - 1-3年工龄：5天
   - 3-5年工龄：10天
   - 5年以上：15天
2. 病假：凭医院证明，每年累计不超过30天
3. 事假：需提前申请，经部门主管批准
""",
        metadata={"type": "人事制度", "title": "请假制度"}
    ),
    Document(
        page_content="""
报销流程：
1. 收集发票：确保发票抬头为公司全称
2. 填写报销单：在OA系统中提交
3. 审批流程：部门主管 → 财务审核 → 出纳付款
4. 到账时间：审批通过后3-5个工作日
5. 注意：单笔超过5000元需总经理审批
""",
        metadata={"type": "财务制度", "title": "报销流程"}
    ),
    Document(
        page_content="""
技术栈规范：
1. 后端：Python + FastAPI（新项目）/ Java + Spring Boot（遗留项目）
2. 前端：Vue3 + TypeScript
3. 数据库：MySQL（关系型）/ Redis（缓存）/ MongoDB（文档型）
4. 部署：Docker + Kubernetes
5. 监控：Prometheus + Grafana
""",
        metadata={"type": "技术规范", "title": "技术栈"}
    ),
    Document(
        page_content="""
代码审查规范：
1. 所有代码合并前必须经过 Code Review
2. 审查重点：
   - 代码逻辑正确性
   - 是否有安全漏洞
   - 性能优化建议
   - 代码风格一致性
3. 审查时间：一般不超过24小时
4. 审查工具：GitLab Merge Request
""",
        metadata={"type": "开发规范", "title": "代码审查"}
    ),
    Document(
        page_content="""
新员工入职流程：
1. 第一天：领取工牌、电脑，开通各系统账号
2. 第一周：完成新员工培训，了解公司文化
3. 第一月：熟悉项目代码，完成第一个任务
4. 试用期：3-6个月，期间有导师一对一指导
""",
        metadata={"type": "人事制度", "title": "入职流程"}
    ),
]

print(f"准备了 {len(documents)} 个文档")
for i, doc in enumerate(documents):
    print(f"  {i+1}. [{doc.metadata['type']}] {doc.metadata['title']}")
print()

# ============================================
# 3. 创建向量数据库
# ============================================
print("=== 3. 创建向量数据库 ===")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./knowledge_db_real"
)

print("✓ 向量数据库创建完成")
print()

# ============================================
# 4. 创建 RAG 链
# ============================================
print("=== 4. 创建 RAG 链 ===")

llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是公司智能助手，根据以下文档内容回答用户问题。
如果文档中没有相关信息，请诚实地说"文档中没有相关信息"。

文档内容：
{context}"""),
    ("user", "{question}")
])

def format_docs(docs):
    return "\n\n".join([f"【{doc.metadata['title']}】\n{doc.page_content}" for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("✓ RAG 链创建完成")
print()

# ============================================
# 5. 测试 RAG
# ============================================
print("=== 5. 测试 RAG 问答 ===")

questions = [
    "我工作3年了，有多少天年假？",
    "报销流程是怎样的？",
    "公司的技术栈是什么？",
    "代码审查需要注意什么？",
    "新员工入职第一天要做什么？",
    "公司食堂在哪里？",
]

for question in questions:
    print(f"\n问题: {question}")
    print("-" * 50)
    
    docs = retriever.invoke(question)
    print(f"检索到 {len(docs)} 个相关文档:")
    for doc in docs:
        print(f"  - {doc.metadata['title']}")
    
    answer = rag_chain.invoke(question)
    print(f"\n回答:\n{answer}")
    print("=" * 50)

print("\n✓ Day 5 RAG 完成！")
