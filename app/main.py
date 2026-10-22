from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import json
import uvicorn

from .agent import ask_stream, resume_stream
from .vector_store import init_vector_store
from .document_loader import load_documents_from_dir
from . import config

#静态资源目录:基于本文件位置解析,避免依赖启动时的工作目录
STATIC_DIR = Path(__file__).resolve().parent / "static"

#多轮对话历史现由 LangGraph MemorySaver checkpointer 管理(按 thread_id=session_id 持久化),
#进程内不再维护 sessions 字典.

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

#定义请求数据模型
class ChatRequest(BaseModel):
    session_id: str
    message: str

#恢复 interrupt 的请求数据模型(Human-in-the-Loop)
class ResumeRequest(BaseModel):
    session_id: str
    resume_value: dict

#根路由:返回前端界面
@app.get("/")
def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

#聊天窗口(SSE 流式):逐 token 推送状态/进度/回答
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    #直接把 session_id 作为 thread_id 透传给 ask_stream,
    #MemorySaver checkpointer 会按 thread_id 自动加载/持久化多轮历史.

    async def event_stream():
        async for evt in ask_stream(message, session_id):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    #返回 SSE 流
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

#恢复 interrupt(SSE 流式):用户在分析确认卡片点确认后调用,继续推送 status/token/done
@app.post("/resume")
async def resume_endpoint(request: ResumeRequest):
    session_id = request.session_id
    resume_value = request.resume_value

    async def event_stream():
        async for evt in resume_stream(session_id, resume_value):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

#用于本地直接运行(非Docker环境)
#注意:不要使用reload=True,Windows+PyCharm下reload会启用子进程模式,
#导致主进程控制台看不到监听日志且子进程实际可能未成功启动.
#本地调试直接用PyCharm断点即可,无需reload.
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
