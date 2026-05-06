# Qdrant 文档切片向量存储

将文档智能切片并存储到 Qdrant 向量数据库，支持语义检索。

## 功能特性

- ✅ 支持多种文档格式（TXT、MD、PDF）
- ✅ 智能切片（可配置大小和重叠）
- ✅ 向量嵌入（支持多种 Embedding 模型）
- ✅ Qdrant 向量存储
- ✅ 语义检索
- ✅ Streamlit 可视化界面

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Qdrant

```bash
# Docker 方式
docker run -p 6333:6333 qdrant/qdrant

# 或使用 Qdrant Cloud
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 使用方式

#### 方式一：命令行

```python
from qdrant_chunker import QdrantDocChunker

# 初始化
chunker = QdrantDocChunker(
    qdrant_host="localhost",
    qdrant_port=6333
)

# 处理文档
result = chunker.process_document(
    file_path="path/to/your/doc.txt",
    collection_name="my_docs",
    chunk_size=500,
    chunk_overlap=50
)

print(result)

# 语义检索
results = chunker.search(
    collection_name="my_docs",
    query="你的查询内容",
    top_k=5
)

for r in results:
    print(f"分数: {r['score']}")
    print(f"内容: {r['content']}")
```

#### 方式二：Streamlit 界面

```bash
streamlit run examples.py
```

## 参数说明

### QdrantDocChunker 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| qdrant_host | str | localhost | Qdrant 服务地址 |
| qdrant_port | int | 6333 | Qdrant 端口 |
| embedding_model | str | text-embedding-v3 | Embedding 模型 |
| api_key | str | 从环境变量获取 | API 密钥 |
| base_url | str | DashScope | API 基础 URL |

### process_document 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file_path | str | 必填 | 文档路径 |
| collection_name | str | 必填 | Qdrant 集合名称 |
| chunk_size | int | 500 | 切片大小（字符） |
| chunk_overlap | int | 50 | 切片重叠（字符） |
| metadata | dict | None | 额外元数据 |

### search 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| collection_name | str | 必填 | 集合名称 |
| query | str | 必填 | 查询文本 |
| top_k | int | 5 | 返回数量 |

## 最佳实践

### 切片大小建议

| 文档类型 | 推荐 chunk_size | 推荐 chunk_overlap |
|----------|-----------------|-------------------|
| 技术文档 | 500-800 | 50-100 |
| 法律条款 | 300-500 | 30-50 |
| 新闻资讯 | 400-600 | 40-60 |
| 小说/长文 | 800-1000 | 100-200 |

### 元数据建议

```python
metadata = {
    "source": "文档来源",
    "version": "文档版本",
    "category": "文档分类",
    "author": "作者",
    "date": "日期"
}
```

## 常见问题

### Q: Qdrant 连接失败？

确保 Qdrant 服务已启动：

```bash
docker run -p 6333:6333 qdrant/qdrant
```

访问 http://localhost:6333/dashboard 确认服务状态。

### Q: Embedding API 报错？

检查 API Key 是否正确配置：

```bash
# .env 文件
DASHSCOPE_API_KEY=sk-xxxxx
```

### Q: PDF 解析失败？

安装 pypdf：

```bash
pip install pypdf
```

### Q: 向量维度不匹配？

不同 Embedding 模型向量维度不同：

| 模型 | 维度 |
|------|------|
| text-embedding-v3 | 1024 |
| text-embedding-v2 | 1536 |
| text-embedding-ada-002 | 1536 |

创建集合时需指定正确的向量维度。

## 许可证

MIT License
