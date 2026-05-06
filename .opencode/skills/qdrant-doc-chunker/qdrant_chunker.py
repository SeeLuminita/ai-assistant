"""
文档切片 + Qdrant向量存储工具
支持PDF、TXT、MD等格式文档的切片和向量化存储
"""

import os
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

load_dotenv()


class QdrantDocChunker:
    """文档切片并存储到Qdrant向量数据库"""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化
        
        Args:
            qdrant_host: Qdrant服务地址
            qdrant_port: Qdrant端口
            embedding_model: Embedding模型名称
            api_key: Embedding API密钥
            base_url: API基础URL（用于兼容模式）
        """
        # 初始化Qdrant客户端
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # 初始化Embedding客户端
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def load_document(self, file_path: str) -> str:
        """加载文档内容"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            # PDF需要pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("请安装pypdf: pip install pypdf")
        
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict]:
        """
        文本切片
        
        Args:
            text: 原始文本
            chunk_size: 切片大小（字符数）
            chunk_overlap: 重叠字符数
        
        Returns:
            切片列表，每个元素包含 content 和 metadata
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": min(end, len(text))
                    }
                })
                chunk_index += 1
            
            start = end - chunk_overlap
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本向量"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        return [item.embedding for item in response.data]
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024
    ):
        """创建Qdrant集合"""
        try:
            self.qdrant.get_collection(collection_name)
            print(f"集合 {collection_name} 已存在")
        except Exception:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"已创建集合: {collection_name}")
    
    def process_document(
        self,
        file_path: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        处理文档：加载 -> 切片 -> 向量化 -> 存储
        
        Args:
            file_path: 文档路径
            collection_name: Qdrant集合名称
            chunk_size: 切片大小
            chunk_overlap: 重叠大小
            metadata: 额外元数据
        
        Returns:
            处理结果
        """
        print(f"正在处理文档: {file_path}")
        
        # 1. 加载文档
        text = self.load_document(file_path)
        print(f"文档长度: {len(text)} 字符")
        
        # 2. 切片
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        print(f"切片数量: {len(chunks)}")
        
        # 3. 创建集合
        self.create_collection(collection_name)
        
        # 4. 批量向量化并存储
        batch_size = 20
        vector_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]
            
            # 获取向量
            embeddings = self.get_embeddings(texts)
            
            # 构建Point
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point_id = str(uuid.uuid4())
                chunk_metadata = {
                    **chunk["metadata"],
                    "content": chunk["content"][:200],  # 存储前200字符预览
                    "file_path": file_path,
                    **(metadata or {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk_metadata
                ))
                vector_ids.append(point_id)
            
            # 存储到Qdrant
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"已处理: {min(i + batch_size, len(chunks))}/{len(chunks)}")
        
        return {
            "status": "success",
            "collection": collection_name,
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "embedding_model": self.embedding_model,
            "vector_ids": vector_ids
        }
    
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        语义检索
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            检索结果
        """
        # 获取查询向量
        query_embedding = self.get_embeddings([query])[0]
        
        # 搜索
        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            }
            for hit in results
        ]


# 使用示例
if __name__ == "__main__":
    # 初始化
    chunker = QdrantDocChunker(
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    # 处理文档
    result = chunker.process_document(
        file_path="E:\\docs\\保险条款.txt",
        collection_name="insurance_docs",
        chunk_size=500,
        chunk_overlap=50,
        metadata={"source": "保险条款", "version": "v1.0"}
    )
    
    print(f"\n处理结果: {result}")
    
    # 语义检索
    results = chunker.search(
        collection_name="insurance_docs",
        query="保险理赔流程是什么",
        top_k=3
    )
    
    print("\n检索结果:")
    for r in results:
        print(f"分数: {r['score']:.4f}")
        print(f"内容: {r['content'][:100]}...")
        print("-" * 50)
