"""
LangChain Day 4: 向量数据库（模拟 Embedding 版）
- 使用 Chroma 向量数据库
- 使用模拟 Embedding（演示用）
- 相似度搜索
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
import numpy as np

load_dotenv()

# ============================================
# 1. 模拟 Embedding 类
# ============================================
class MockEmbeddings(Embeddings):
    """模拟 Embedding（演示用，实际项目用真实模型）"""
    
    def __init__(self):
        # 关键词向量映射（简化版）
        self.keyword_vectors = {
            "python": [1.0, 0.0, 0.0, 0.0, 0.0],
            "java": [0.9, 0.1, 0.0, 0.0, 0.0],
            "javascript": [0.85, 0.15, 0.0, 0.0, 0.0],
            "机器学习": [0.0, 0.0, 1.0, 0.0, 0.0],
            "深度学习": [0.0, 0.0, 0.9, 0.1, 0.0],
            "人工智能": [0.0, 0.0, 0.95, 0.05, 0.0],
            "框架": [0.0, 0.0, 0.0, 1.0, 0.0],
            "fastapi": [0.8, 0.0, 0.0, 0.9, 0.0],
            "langchain": [0.7, 0.0, 0.0, 0.85, 0.0],
            "向量": [0.0, 0.0, 0.0, 0.0, 1.0],
            "数据库": [0.0, 0.0, 0.0, 0.0, 0.9],
        }
    
    def _text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量"""
        text_lower = text.lower()
        vector = [0.0] * 5
        
        # 根据关键词计算向量
        for keyword, kv in self.keyword_vectors.items():
            if keyword in text_lower:
                for i in range(5):
                    vector[i] = max(vector[i], kv[i])
        
        # 如果没有匹配的关键词，生成随机向量
        if sum(vector) == 0:
            np.random.seed(hash(text) % (2**32))
            vector = np.random.rand(5).tolist()
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量转换文档"""
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """转换单个查询"""
        return self._text_to_vector(text)

# ============================================
# 2. 什么是 Embedding
# ============================================
print("=== 1. Embedding 概念 ===")
print("""
Embedding = 将文本转换为数字向量

例如：
"Python编程语言" → [1.0, 0.0, 0.0, 0.0, 0.0]
"机器学习技术"   → [0.0, 0.0, 1.0, 0.0, 0.0]
"FastAPI框架"   → [0.8, 0.0, 0.0, 0.9, 0.0]

相似含义的文本 → 向量距离更近
""")
print()

# ============================================
# 3. 初始化 Embedding
# ============================================
print("=== 2. 初始化 Embedding ===")
embeddings = MockEmbeddings()
print("✓ Embedding 模型加载完成（模拟版）")
print()

# ============================================
# 4. 准备文档数据
# ============================================
print("=== 3. 准备文档数据 ===")

documents = [
    Document(page_content="Python是一种高级编程语言，以简洁易读著称。广泛应用于Web开发、数据分析、人工智能等领域。", metadata={"topic": "python", "type": "编程语言"}),
    Document(page_content="Java是一种面向对象的编程语言，具有跨平台、安全性高等特点。广泛用于企业级应用开发。", metadata={"topic": "java", "type": "编程语言"}),
    Document(page_content="JavaScript是一种脚本语言，主要用于Web前端开发，也可以用于后端开发(Node.js)。", metadata={"topic": "javascript", "type": "编程语言"}),
    Document(page_content="机器学习是人工智能的一个分支，让计算机从数据中自动学习和改进，无需明确编程。", metadata={"topic": "ml", "type": "AI技术"}),
    Document(page_content="深度学习是机器学习的子领域，使用多层神经网络处理图像、文本、语音等复杂数据。", metadata={"topic": "dl", "type": "AI技术"}),
    Document(page_content="FastAPI是一个现代、高性能的Python Web框架，支持异步编程和自动API文档生成。", metadata={"topic": "fastapi", "type": "框架"}),
    Document(page_content="LangChain是一个开发LLM应用的框架，提供链式调用、Agent、向量检索等功能。", metadata={"topic": "langchain", "type": "框架"}),
    Document(page_content="向量数据库用于存储文本的向量表示，支持相似度搜索，常用于RAG应用。", metadata={"topic": "vector_db", "type": "技术"}),
]

print(f"准备了 {len(documents)} 个文档")
for i, doc in enumerate(documents):
    print(f"  {i+1}. [{doc.metadata['type']}] {doc.page_content[:25]}...")
print()

# ============================================
# 5. 创建 Chroma 向量数据库
# ============================================
print("=== 4. 创建向量数据库 ===")

# 创建并持久化向量数据库
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✓ 向量数据库创建完成，已保存到 ./chroma_db")
print()

# ============================================
# 6. 相似度搜索
# ============================================
print("=== 5. 相似度搜索 ===")

queries = [
    ("编程语言有哪些？", 3),
    ("人工智能和机器学习", 2),
    ("Python的Web框架", 2),
    ("什么是向量数据库", 2),
]

for query, k in queries:
    print(f"\n查询: {query}")
    results = vectorstore.similarity_search(query, k=k)
    print(f"最相关的 {k} 个结果:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. [{doc.metadata['topic']}] {doc.page_content[:40]}...")

print()

# ============================================
# 7. 带分数的相似度搜索
# ============================================
print("=== 6. 带分数的相似度搜索 ===")

query = "深度学习神经网络"
print(f"查询: {query}")
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print("结果（分数越低越相似）:")
for doc, score in results_with_scores:
    print(f"  分数: {score:.4f} | {doc.page_content[:35]}...")

print()

# ============================================
# 8. 添加新文档
# ============================================
print("=== 7. 动态添加文档 ===")

new_doc = Document(
    page_content="Streamlit是一个Python库，用于快速构建数据科学和机器学习Web应用。",
    metadata={"topic": "streamlit", "type": "框架"}
)

vectorstore.add_documents([new_doc])
print("✓ 已添加新文档: Streamlit")

# 测试新添加的文档
query = "Python快速构建Web应用"
results = vectorstore.similarity_search(query, k=2)
print(f"\n查询: {query}")
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.page_content[:40]}...")

print("\n" + "="*60)
print("Day 4 完成！你已掌握：")
print("  1. Embedding 概念 - 文本转向量")
print("  2. Chroma 向量数据库 - 存储和检索")
print("  3. 相似度搜索 - 找到最相关的内容")
print()
print("⚠️ 注意：当前使用模拟 Embedding（演示用）")
print("   实际项目推荐使用：")
print("   - OpenAI Embeddings（需API）")
print("   - 本地模型（需下载，使用镜像：https://hf-mirror.com）")
print()
print("下一步：Day 5 将学习 RAG（检索增强生成）")
print("  结合向量数据库 + LLM 实现智能问答")
