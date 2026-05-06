"""
Streamlit应用：文档切片向量存储
支持上传文档，切片后存入Qdrant向量数据库
"""

import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_chunker import QdrantDocChunker
import tempfile

load_dotenv()

st.set_page_config(page_title="文档切片向量存储", page_icon="📚", layout="wide")

st.title("📚 文档切片向量存储")
st.write("上传文档，智能切片并存入Qdrant向量数据库")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # Qdrant配置
    qdrant_host = st.text_input("Qdrant地址", value="localhost")
    qdrant_port = st.number_input("Qdrant端口", value=6333)
    
    # Embedding配置
    embedding_model = st.selectbox(
        "Embedding模型",
        ["text-embedding-v3", "text-embedding-v2", "text-embedding-ada-002"]
    )
    
    st.divider()
    
    # 切片配置
    chunk_size = st.slider("切片大小（字符）", 100, 1000, 500)
    chunk_overlap = st.slider("切片重叠（字符）", 0, 200, 50)
    
    st.divider()
    
    # 元数据
    doc_source = st.text_input("文档来源", placeholder="例如：产品手册")
    doc_version = st.text_input("文档版本", placeholder="例如：v1.0")

# 主区域
tab1, tab2 = st.tabs(["📤 文档入库", "🔍 语义检索"])

with tab1:
    st.subheader("上传文档")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择文档",
        type=["txt", "md", "pdf"],
        help="支持TXT、MD、PDF格式"
    )
    
    # 集合名称
    collection_name = st.text_input(
        "Qdrant集合名称",
        placeholder="例如：insurance_docs"
    )
    
    # 处理按钮
    if st.button("🚀 开始处理", type="primary"):
        if not uploaded_file:
            st.warning("请上传文档")
        elif not collection_name:
            st.warning("请输入集合名称")
        else:
            # 保存上传文件到临时目录
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                with st.spinner("处理中..."):
                    # 初始化chunker
                    chunker = QdrantDocChunker(
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        embedding_model=embedding_model
                    )
                    
                    # 处理文档
                    metadata = {}
                    if doc_source:
                        metadata["source"] = doc_source
                    if doc_version:
                        metadata["version"] = doc_version
                    
                    result = chunker.process_document(
                        file_path=tmp_path,
                        collection_name=collection_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        metadata=metadata if metadata else None
                    )
                
                # 显示结果
                st.success("✅ 处理完成！")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("切片数量", result["total_chunks"])
                with col2:
                    st.metric("切片大小", result["chunk_size"])
                with col3:
                    st.metric("向量维度", "1024")
                
                with st.expander("📋 详细信息"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"处理失败: {str(e)}")
            finally:
                # 清理临时文件
                os.unlink(tmp_path)

with tab2:
    st.subheader("语义检索")
    
    # 检索配置
    search_collection = st.text_input("集合名称", placeholder="例如：insurance_docs")
    search_query = st.text_area("查询内容", placeholder="输入要检索的问题...", height=100)
    top_k = st.slider("返回数量", 1, 20, 5)
    
    if st.button("🔍 检索", type="primary"):
        if not search_collection:
            st.warning("请输入集合名称")
        elif not search_query:
            st.warning("请输入查询内容")
        else:
            try:
                with st.spinner("检索中..."):
                    chunker = QdrantDocChunker(
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        embedding_model=embedding_model
                    )
                    
                    results = chunker.search(
                        collection_name=search_collection,
                        query=search_query,
                        top_k=top_k
                    )
                
                if results:
                    st.success(f"找到 {len(results)} 条相关结果")
                    
                    for i, r in enumerate(results):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric("相似度", f"{r['score']:.4f}")
                            with col2:
                                st.markdown(f"**内容预览:**")
                                st.text(r['content'][:300] + "..." if len(r['content']) > 300 else r['content'])
                            
                            with st.expander("查看元数据"):
                                st.json(r['metadata'])
                            
                            st.divider()
                else:
                    st.info("未找到相关结果")
                    
            except Exception as e:
                st.error(f"检索失败: {str(e)}")

# 页脚
st.divider()
st.caption("💡 提示：确保Qdrant服务已启动，Embedding API已配置")
