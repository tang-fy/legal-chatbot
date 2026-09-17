from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List
from langchain_core.messages import BaseMessage
import uvicorn

from .agent import ask
from .vector_store import init_vector_store
from .document_loader import load_documents_from_dir
from . import config

#静态资源目录:基于本文件位置解析,避免依赖启动时的工作目录
STATIC_DIR = Path(__file__).resolve().parent / "static"

#存储所有会话的历史记录, key为session_id
sessions: Dict[str, List[BaseMessage]] = {}

#定义生命周期管理:在应用启动时初始化向量库
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在初始化法律文档向量库...")
    #加载文档目录中的所有文档
    docs = load_documents_from_dir(config.DOCUMENTS_DIR)
    #初始化Milvus,如果集合为空则插入文档
    init_vector_store(docs)
    print("向量库初始化完成!")
    yield #应用运行期间

#创建FastAPI应用,并传入lifespan
app = FastAPI(lifespan=lifespan)

#挂载静态文件目录,用于前端界面
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

#定义请求和响应的数据模型
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

#根路由:返回前端界面
@app.get("/")
def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

#聊天窗口
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    #获取该会话的历史记录,如果没有则新建空列表
    history = sessions.get(session_id, [])
    #调用核心回答函数
    answer, updated_history = ask(message, history)
    #更新会话历史
    sessions[session_id] = updated_history
    return ChatResponse(reply=answer)

#用于本地直接运行(非Docker环境)
#注意:不要使用reload=True,Windows+PyCharm下reload会启用子进程模式,
#导致主进程控制台看不到监听日志且子进程实际可能未成功启动.
#本地调试直接用PyCharm断点即可,无需reload.
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)