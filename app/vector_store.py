import time
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from typing import List, Optional
from langchain_core.documents import Document
from .embeddings import DirectOllamaEmbeddings
from . import config

_embeddings = DirectOllamaEmbeddings(
    model=config.EMBED_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    batch_size=config.EMBED_BATCH_SIZE,
)

_vector_store: Optional[Milvus] = None


def get_vector_store(max_retries: int = 30, delay: int = 5) -> Milvus:
    """获取 Milvus 连接，带重试机制。"""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[VectorStore] 尝试连接 Milvus (第 {attempt}/{max_retries} 次): "
                  f"host={config.MILVUS_HOST}, port={config.MILVUS_PORT}")
            _vector_store = Milvus(
                embedding_function=_embeddings,
                collection_name=config.MILVUS_COLLECTION,
                # pymilvus 3.x 的 MilvusClient 只接受 uri,
                # 传 host/port 会被静默忽略并回退到 localhost
                connection_args={
                    "uri": f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
                },
            )
            print("[VectorStore] Milvus 连接成功！")
            return _vector_store
        except Exception as e:
            last_err = e
            print(f"[VectorStore] 连接失败: {e}，{delay} 秒后重试...")
            time.sleep(delay)

    raise RuntimeError(f"无法连接 Milvus（重试 {max_retries} 次后失败）: {last_err}")


def _collection_has_data() -> bool:
    """用 pymilvus 独立检查 collection 是否已有数据。"""
    client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
    try:
        if not client.has_collection(config.MILVUS_COLLECTION):
            print(f"[VectorStore] Collection '{config.MILVUS_COLLECTION}' 不存在，视为空")
            return False
        # pymilvus 3.x / Milvus 3.0 中 get_collection_stats() 的 row_count 不可靠
        # (数据未 flush 时恒返回 0), 改用 count(*) 聚合查询获取真实行数,
        # 否则应用每次启动都会误判为空库并重复插入全部文档
        rows = client.query(
            config.MILVUS_COLLECTION,
            filter="",
            output_fields=["count(*)"],
        )
        row_count = rows[0]["count(*)"] if rows else 0
        print(f"[VectorStore] Collection 现有 {row_count} 条记录")
        return row_count > 0
    except Exception as e:
        print(f"[VectorStore] 检查 collection 时出错: {e}，视为空")
        return False
    finally:
        client.close()


def init_vector_store(documents: List[Document] = None) -> Milvus:
    store = get_vector_store()

    if not documents:
        return store

    if _collection_has_data():
        print("[VectorStore] Collection 已有数据，跳过插入")
        return store

    print("[VectorStore] Collection 为空，插入文档...")
    try:
        store.add_documents(documents)
        print(f"[VectorStore] 已插入 {len(documents)} 个文档")
    except Exception as e:
        print(f"[VectorStore] 插入文档失败: {e}")
    return store


def similarity_search(query: str, k: int = config.RAG_TOP_K) -> List[Document]:
    """向量检索,返回最相关的 Top-K 文档.

    优化:
    1. 多召回 2 倍候选(2*k),再用相似度分数过滤掉不相关的
    2. 过滤掉分数过低的结果(相关性阈值),避免问房屋租赁却返回香港基本法
    3. 最终只返回 Top-K
    """
    store = get_vector_store()
    # 多召回候选,给过滤留余量
    candidates = store.similarity_search_with_score(query, k=k * 2)
    if not candidates:
        return []

    # 打印分数分布,便于调参
    scores = [round(score, 4) for _, score in candidates]
    print(f"[VectorStore] 检索分数分布: min={min(scores)}, max={max(scores)}, mean={sum(scores)/len(scores):.4f}")

    # 过滤掉相关性过低的结果
    # Milvus similarity_search_with_score 返回的是 L2 距离(越小越相似)
    # 阈值 1.0: L2 距离 > 1.0 认为不相关
    threshold = 1.0
    filtered = [(doc, score) for doc, score in candidates if score <= threshold]
    if not filtered:
        print(f"[VectorStore] 所有候选分数均 > {threshold},无相关文档")
        return []
    # 最终只取 Top-K
    filtered = filtered[:k]
    print(f"[VectorStore] 召回 {len(candidates)} 条,过滤后保留 {len(filtered)} 条")
    return [doc for doc, _ in filtered]