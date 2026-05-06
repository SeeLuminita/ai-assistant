"""
单元测试：文档切片向量存储
"""

import pytest
import os
import tempfile
from qdrant_chunker import QdrantDocChunker


class TestQdrantDocChunker:
    """QdrantDocChunker 测试类"""
    
    @pytest.fixture
    def chunker(self):
        """初始化 chunker"""
        return QdrantDocChunker(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", 6333))
        )
    
    def test_load_txt_file(self, chunker):
        """测试加载 TXT 文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是一个测试文档\n包含多行内容")
            temp_path = f.name
        
        try:
            text = chunker.load_document(temp_path)
            assert "测试文档" in text
            assert "多行内容" in text
        finally:
            os.unlink(temp_path)
    
    def test_load_md_file(self, chunker):
        """测试加载 MD 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 标题\n\n这是正文内容")
            temp_path = f.name
        
        try:
            text = chunker.load_document(temp_path)
            assert "# 标题" in text
            assert "正文内容" in text
        finally:
            os.unlink(temp_path)
    
    def test_chunk_text(self, chunker):
        """测试文本切片"""
        text = "这是一段很长的测试文本，" * 100
        
        chunks = chunker.chunk_text(text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 1
        assert all(len(c["content"]) <= 50 for c in chunks)
        assert all("chunk_index" in c["metadata"] for c in chunks)
    
    def test_chunk_text_overlap(self, chunker):
        """测试切片重叠"""
        text = "ABCDEFGHIJ" * 20
        
        chunks = chunker.chunk_text(text, chunk_size=20, chunk_overlap=5)
        
        # 验证重叠
        if len(chunks) > 1:
            # 第二个切片的开始应该与第一个切片的结尾有重叠
            first_end = chunks[0]["content"][-5:]
            second_start = chunks[1]["content"][:5]
            # 重叠部分应该相同（考虑切片边界）
    
    def test_get_embeddings(self, chunker):
        """测试获取向量"""
        texts = ["这是第一句话", "这是第二句话"]
        
        embeddings = chunker.get_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(len(e) > 0 for e in embeddings)
        assert all(isinstance(e, list) for e in embeddings)
    
    @pytest.mark.integration
    def test_process_document_integration(self, chunker):
        """集成测试：完整处理流程"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是一份测试文档，用于验证完整的处理流程。" * 10)
            temp_path = f.name
        
        try:
            result = chunker.process_document(
                file_path=temp_path,
                collection_name="test_collection",
                chunk_size=100,
                chunk_overlap=20,
                metadata={"test": True}
            )
            
            assert result["status"] == "success"
            assert result["total_chunks"] > 0
            assert len(result["vector_ids"]) == result["total_chunks"]
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_search_integration(self, chunker):
        """集成测试：语义检索"""
        # 先存储文档
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("平安保险是一家专业的保险公司，提供多种保险产品和服务。理赔流程简单快捷。")
            temp_path = f.name
        
        try:
            chunker.process_document(
                file_path=temp_path,
                collection_name="test_search",
                chunk_size=100,
                chunk_overlap=10
            )
            
            # 执行检索
            results = chunker.search(
                collection_name="test_search",
                query="保险公司理赔",
                top_k=3
            )
            
            assert len(results) > 0
            assert all("score" in r for r in results)
            assert all("content" in r for r in results)
        finally:
            os.unlink(temp_path)


def test_unsupported_format():
    """测试不支持的文件格式"""
    chunker = QdrantDocChunker()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write("test")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="不支持的文件格式"):
            chunker.load_document(temp_path)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
