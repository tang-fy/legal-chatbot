import os
from typing import List
# 导入LangChain社区提供的各种文档加载器
# 注意: UnstructuredMarkdownLoader 依赖较重的 unstructured 包,
# 改为在实际遇到 .md 文件时再惰性导入,避免缺少该依赖时整个应用无法启动
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from . import config

def load_documents_from_dir(
    dir_path: str = config.DOCUMENTS_DIR,
) -> List[Document]:
    """
    从指定目录加载所有支持的文档,并进行文本分块.
    返回一个LangChain Document 列表,每个Document包含page_content和metadata.
    """
    # 定义文件扩展名与对应加载器的映射
    loaders = {
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".docx": (Docx2txtLoader, {}),
    }
    raw_docs = [] #存储原始加载的文档(未分块)

    #检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"[Loader]目录不存在: {dir_path}")
        return raw_docs

    #遍历目录下所有文件
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if not os.path.isfile(filepath): #跳过子目录
            continue
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".md":
            # .md 加载器依赖 unstructured,仅在需要时惰性导入
            try:
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                loader_cls, kwargs = UnstructuredMarkdownLoader, {}
            except Exception as e:
                print(f"[Loader]加载{filename}失败: 缺少 unstructured 依赖({e})")
                continue
        elif ext in loaders:
            loader_cls, kwargs = loaders[ext]
        else:
            continue
        try:
            loader = loader_cls(filepath, **kwargs)
            loaded = loader.load()
            # 给每个文档加上来源标记(文件名),便于后续检索时引用溯源
            for doc in loaded:
                doc.metadata["source"] = filename
            raw_docs.extend(loaded)
            print(f"[Loader]已加载: {filename} ({len(loaded)}个文档)")
        except Exception as e:
            print(f"[Loader]加载{filename}失败: {e}")
    if not raw_docs:
        print("[Loader]未找到任何可加载的文档")
        return []

    #创建文本分割器,按字符递归分割,chunk_size和chunk_overlap在config中定义
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    #将原始文档切分成较小的块,便于向量化和检索
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"[Loader]共加载{len(raw_docs)}个原始文档,分块后得到{len(split_docs)}个chunks")
    return split_docs