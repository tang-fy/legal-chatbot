import time
import requests
from typing import List
from langchain_core.embeddings import Embeddings

class DirectOllamaEmbeddings(Embeddings):
    """
    直接调用Ollama HTTP API 的E吗bedding类.
    为什么不使用官方的SDK? 因为官方的SDK在Windows下存在启动tokenizer 子进程失败的问题
    这里直接构造HTTP请求,绕过该问题,同时支持批量处理提升效率.
    """
    def __init__(self, model: str, base_url: str, batch_size: int=32, timeout: int=120):
        self.model = model #嵌入模型名称
        self.base_url = base_url.rstrip('/') #去除末尾斜杠,保证url拼接正确
        self.embed_url = f"{self.base_url}/api/embed" #Ollama嵌入接口
        self.batch_size = batch_size  #每次发送的文本数量
        self.timeout = timeout  #请求超时时间(秒)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        内部方法:向Ollama发送一批文本,获取对应的向量列表.
        Ollama的/api/embed接口接受{"model": ..., "input": [text1, text2, ...]},
        返回{"embeddings": [[...], [...]. ...]}.
        """
        resp = requests.post(
            self.embed_url,
            json={"model": self.model, "input": texts},
            timeout=self.timeout
        )
        resp.raise_for_status() #如果状态码非2xx,抛出异常
        data = resp.json()
        return data["embeddings"] #返回向量列表

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain要求的接口:批量将文档文本转换为向量.
        内部按batch_size分块,逐个批次调用_embed_batch,并打印进度.
        """
        if not texts:
            return []
        print(f"[Embed]共{len(texts)}个chunks,每批{self.batch_size}个...")
        results = []
        start = time.time()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) - 1) // self.batch_size + 1
            print(f"[Embed]批次{batch_num}/{total_batches}({len(batch)}个)...", end=" ")
            try:
                emb = self._embed_batch(batch)
                results.extend(emb)
                print("成功")
            except Exception as e:
                print(f"失败:{e}")
                raise
        print(f"[Embed]完成,耗时{time.time()-start:.1f}s")
        return results

    def embed_query(self, texts: str) -> List[float]:
        """
        LangChain要求的接口:将单个查询文本转换为向量.
        直接调用_embed_batch,但只传一个元素.
        """
        return self.embed_documents([texts])[0]
