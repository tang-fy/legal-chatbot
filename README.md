# ⚖️ 法律咨询智能问答 Agent

基于 **RAG + LLM** 的法律咨询问答平台，面向普通用户输出"**行为定性 → 分情形法律后果 → 可执行行动建议 → 法律依据**"的结构化法律分析，而不是机械罗列法条。

内置 59 部现行法律（宪法、民法典、刑法、民事诉讼法、道路交通安全法等）、917+ 向量片段，支持多轮对话与法律案例库自然语言查询。

---

## ✨ 功能特性

- **结构化法律分析**：每次回答按"行为分析 / 法律后果（分情形）/ 行动建议 / 法律依据 / 免责声明"五段式输出，普通人也能看懂
- **双链路智能路由**
  - **Fast Path**：普通法律咨询 → 单次 LLM 调用直连 RAG，低延迟
  - **Agent 工具链**：案例检索/统计类问题 → ReAct Agent 自主调用"法条检索 + NL2SQL 查 MySQL 案例库"
- **RAG 检索增强**：法条完整性分块（1200 字符）+ BGE-M3 向量化 + Milvus 检索 + L2 距离阈值过滤无关召回 + 来源法条溯源
- **NL2SQL 案例查询**：自然语言直接查询结构化案例库，自动清洗 LLM 生成 SQL 中的 Markdown 包裹
- **稳定的结构化输出**：PydanticOutputParser 约束 JSON Schema，解析失败自动降级兜底，服务不中断
- **现代化 Web 界面**：气泡式对话框、"思考中"加载动画、多轮会话、Enter 发送 / Shift+Enter 换行

---

## 🧱 技术架构

```
┌──────────────────────────────────────────────────────────┐
│                    浏览器 (index.html)                     │
└───────────────────────────┬──────────────────────────────┘
                            │ HTTP /chat
┌───────────────────────────▼──────────────────────────────┐
│                    FastAPI (main.py)                      │
│              lifespan 启动装载 / 会话管理                  │
└───────────────────────────┬──────────────────────────────┘
                            │ ask(question, history)
              ┌─────────────▼──────────────┐
              │        意图路由 (agent.py)   │
              │   正则关键词 + 问题意图判断   │
              └──────┬───────────────┬──────┘
        普通咨询     │               │  案例/统计查询
        ┌────────────▼────┐   ┌──────▼──────────────┐
        │  Fast Path       │   │  ReAct Agent         │
        │  RAG → LLM(1次)  │   │  自主选择工具:        │
        └───────┬─────────┘   │  · search_legal_docs │
                │             │  · query_case (NL2SQL)│
                │             └──────────┬───────────┘
        ┌───────▼─────────────────────────▼──────┐
        │  DeepSeek-V3 (OpenAI 兼容 API) 生成回答  │
        └───────┬─────────────────────────┬──────┘
                │                         │
   ┌────────────▼────────┐    ┌───────────▼───────────┐
   │  Milvus 向量库       │    │  MySQL 法律案例库      │
   │  BGE-M3 嵌入(Ollama) │    │  (NL2SQL)             │
   └─────────────────────┘    └───────────────────────┘
```

### 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python、FastAPI、LangChain、Pydantic v2 |
| 大模型 | DeepSeek-V3（OpenAI 兼容 API） |
| 嵌入模型 | BGE-M3（Ollama 本地部署） |
| 向量数据库 | Milvus 3.x（etcd + MinIO 依赖） |
| 关系数据库 | MySQL 8.0（法律案例库） |
| 基础设施 | Docker Compose |

---

## 📁 项目结构

```
legal-chatbot/
├── app/
│   ├── main.py             # FastAPI 入口、lifespan、/chat 接口
│   ├── agent.py            # 意图路由、Fast Path、ReAct Agent、提示词
│   ├── tools.py            # @tool 封装:法条检索 / NL2SQL 案例查询
│   ├── vector_store.py     # Milvus 初始化、相似度检索、阈值过滤
│   ├── embeddings.py       # Ollama BGE-M3 嵌入封装
│   ├── document_loader.py  # docx/pdf/md 文档加载与分块
│   ├── output_parser.py    # 五段式回答 Pydantic Schema
│   ├── config.py           # 全部环境变量配置
│   └── static/index.html   # 前端聊天界面
├── data/documents/         # 59 部现行法律 docx(随仓库分发)
├── init.sql                # MySQL 案例库建表脚本
├── docker-compose.yml      # etcd/minio/milvus/mysql/ollama 编排
├── start_app.ps1           # Windows 本地启动脚本
└── requirements.txt
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker Desktop（运行基础设施）
- [DeepSeek API Key](https://platform.deepseek.com/)（用于 LLM 推理；新用户有免费额度）

### 1. 克隆项目

```bash
git clone https://github.com/tang-fy/legal-chatbot.git
cd legal-chatbot
```

### 2. 启动基础设施

```bash
docker compose up -d etcd minio milvus mysql ollama ollama-init
```

`ollama-init` 会自动拉取 `bge-m3` 嵌入模型（首次需要几分钟）。等待 Milvus 状态变为 healthy：

```bash
docker compose ps
```

### 3. 安装 Python 依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. 配置环境变量

| 变量名 | 说明 | 示例值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | **必填**，DeepSeek API 密钥 | `sk-xxxxxxxx` |
| `MILVUS_HOST` | 本地运行填 localhost | `localhost` |
| `MYSQL_HOST` | MySQL 地址 | `localhost` |
| `MYSQL_PASSWORD` | 与 docker-compose 中一致 | `123456` |
| `OLLAMA_BASE_URL` | Ollama 地址（嵌入模型用） | `http://localhost:11434` |

PowerShell 设置示例：

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
$env:MILVUS_HOST="localhost"
```

> PyCharm 用户：在 Run Configuration → Environment variables 中配置上述变量，运行方式选择 **Module name = `uvicorn`**，参数填 `app.main:app --host 0.0.0.0 --port 8000`。

### 5. 启动应用

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

首次启动会自动将 `data/documents/` 下的法律文档分块、向量化并写入 Milvus（约 5-15 分钟，之后启动秒级完成）。

浏览器访问 **http://localhost:8000** 即可使用。

---

## 💡 使用示例

**普通咨询（Fast Path）**

> 我在马路上开车撞死了人会被判几年？

回答将按五段式输出：行为定性（涉嫌交通肇事罪，关键看责任划分）→ 分情形后果（不构成犯罪 / 3 年以下 / 3-7 年 / 7 年以上 / 民事赔偿）→ 行动建议（停车报警、切勿逃逸、赔偿争取谅解）→ 法律依据（《刑法》第 133 条）→ 免责声明。

**案例库查询（Agent + NL2SQL）**

> 数据库里一共有多少起交通事故纠纷案件？

Agent 自动生成并执行 SQL，返回统计结果。

也可以直接调用 API：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "message": "租房期间房东把房子卖了,我必须搬走吗?"}'
```

---

## 🔧 关键设计说明

- **为什么用 Fast Path + Agent 双链路？**
  ReAct Agent 每次回答需要 3-4 次 LLM 调用（思考→选工具→观察→回答）。绝大多数普通咨询只需"检索 + 生成"，Fast Path 把 LLM 调用降为 1 次；只有案例库统计这类必须操作结构化数据的请求才走 Agent。

- **Token 预算管控**
  系统提示词（约 1300 tokens）+ 3 条检索片段（约 3500 tokens）+ 输出预留（2048）必须小于模型上下文窗口。曾因预算超限导致系统指令被静默截断、模型退化为机械罗列法条，通过 `TOP_K=3` + 窗口核算修复。

- **法条完整性分块**
  chunk_size 设为 1200 字符（而非常见的 512/800），避免单条法律被切断；重叠 200 字符保证上下文连贯。

---

## ⚠️ 免责声明

本项目仅供学习与技术交流，所有 AI 生成内容**不构成法律意见**。真实法律问题请咨询执业律师。

---

## 📄 License

MIT

## 👤 作者

**唐富焱**（[GitHub: tang-fy](https://github.com/tang-fy)）
