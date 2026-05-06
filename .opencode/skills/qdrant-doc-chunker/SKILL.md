# Qdrant文档切片向量存储

## 描述
将文档进行智能切片，生成向量嵌入，并存储到Qdrant向量数据库中，支持后续的语义检索。

## 触发词
- 文档切片
- 向量化存储
- qdrant
- 文档入库
- 文档embedding
- 切片存入向量库
- 文档向量化

## 使用场景
- 将PDF/TXT/MD等文档切片后存入Qdrant
- 构建RAG知识库
- 文档语义检索准备

## 前置要求
- Qdrant服务已启动（本地或云端）
- Embedding模型API可用（如OpenAI、DashScope等）

## 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file_path | string | 是 | 待处理的文档路径 |
| collection_name | string | 是 | Qdrant集合名称 |
| chunk_size | int | 否 | 切片大小，默认500字符 |
| chunk_overlap | int | 否 | 切片重叠，默认50字符 |
| embedding_model | string | 否 | embedding模型，默认text-embedding-v3 |

## 工作流程

1. **文档加载**：读取指定路径的文档内容
2. **文档切片**：按chunk_size进行切片，支持overlap
3. **生成向量**：调用Embedding API生成向量
4. **存储向量**：将向量和元数据存入Qdrant
5. **返回结果**：切片数量、向量ID列表

## 示例用法

```
请将 E:\docs\保险条款.pdf 切片后存入Qdrant，集合名称为 insurance_docs
```

```
把 E:\knowledge\产品手册.md 向量化存储到 qdrant 的 product_manual 集合，切片大小800
```

## 输出格式

```json
{
  "status": "success",
  "collection": "insurance_docs",
  "total_chunks": 125,
  "chunk_size": 500,
  "embedding_model": "text-embedding-v3",
  "vector_ids": ["uuid1", "uuid2", ...]
}
```

## 注意事项
- 确保Qdrant服务可访问
- 大文档处理时间较长，请耐心等待
- 建议chunk_size设置在300-800之间
- chunk_overlap建议设置为chunk_size的10%
