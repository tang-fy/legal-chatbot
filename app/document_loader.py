import os
import re
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

# 法条标题行:"第二百六十六条" / "第1077条" / "第二十条之一"
_ARTICLE_HEAD_RE = re.compile(
    r"^\s*第[零一二三四五六七八九十百千0-9]+条(?:之[一二三四五六七八九十]+)?"
)
# 编/章/节标题:切块时挂到下一段,保证条文不丢失章节上下文
_HEADING_RE = re.compile(r"^\s*第[零一二三四五六七八九十百]+[编章节]")
# 文件名里的发布日期后缀,拼标题时去掉
_DATE_SUFFIX_RE = re.compile(r"_\d{8}$")


def _law_title(filename: str) -> str:
    """由文件名得到法律名称,如 中华人民共和国刑法_20201226.docx -> 中华人民共和国刑法."""
    stem = os.path.splitext(filename)[0]
    return _DATE_SUFFIX_RE.sub("", stem)


def _split_by_article(doc: Document) -> List[Document]:
    """按"第X条"切分单个法律文档.

    原因:按固定字符数切分会把条文从中间截断,导致检索命中法律文件却找不到条文,
    回答只能凭模型记忆引用法条(评测中的"引用未落地率").
    每个条文独立成块,超过 CHUNK_SIZE 的长条文再按字符递归切分.
    """
    title = _law_title(doc.metadata.get("source", ""))
    segments: List[str] = []
    cur: List[str] = []

    for line in doc.page_content.splitlines():
        # 章/节标题:先把上一段收尾,标题与后续条文放一起
        if _HEADING_RE.match(line):
            if any(_ARTICLE_HEAD_RE.match(x) for x in cur):
                segments.append("\n".join(cur).strip())
                cur = []
            cur.append(line)
            continue
        # 新条文:前一段(若有内容)收尾
        if _ARTICLE_HEAD_RE.match(line) and cur:
            segments.append("\n".join(cur).strip())
            cur = [line]
            continue
        cur.append(line)
    if cur:
        segments.append("\n".join(cur).strip())

    # 没有识别出任何条文时退化为普通字符切分
    if not any(_ARTICLE_HEAD_RE.match(s) for s in segments):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        return splitter.split_documents([doc])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks: List[Document] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        head = _ARTICLE_HEAD_RE.match(seg)
        article = head.group(0).strip() if head else ""
        metadata = dict(doc.metadata)
        metadata["article"] = article
        # 在法律名写进正文,便于"劳动合同法 试用期"这类带法律名的问法召回
        body = f"《{title}》\n{seg}"
        if len(body) <= config.CHUNK_SIZE:
            chunks.append(Document(page_content=body, metadata=metadata))
        else:
            for sub in splitter.split_text(seg):
                chunks.append(Document(page_content=f"《{title}》\n{sub}", metadata=metadata))
    return chunks


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

    # 优先按"第X条"切分,保证每个条文完整;无法识别条文时自动退化为字符切分
    split_docs: List[Document] = []
    for doc in raw_docs:
        split_docs.extend(_split_by_article(doc))
    print(f"[Loader]共加载{len(raw_docs)}个原始文档,按条文切分后得到{len(split_docs)}个chunks")
    return split_docs