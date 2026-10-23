# ⚖️ 法律咨询智能问答 Agent

基于 **LangGraph 多 Agent 编排 + RAG + LLM** 的法律咨询问答平台，面向普通用户输出"**行为定性 → 分情形法律后果 → 可执行行动建议 → 法律依据**"的结构化法律分析，而不是机械罗列法条。

内置 59 部现行法律（宪法、民法典、刑法、民事诉讼法、道路交通安全法等）、7912 个条文级片段（按“第 X 条”切分），支持**多轮对话记忆**、**流式 SSE 输出**、**Human-in-the-Loop 人工介入**与法律案例库自然语言查询。

---

## ✨ 功能特性

- **LangGraph 多 Agent 编排**：基于状态图（StateGraph）的显式节点编排，supervisor 条件路由 → Fast Path 直答 或 Agent 分支并行处理
- **并行 fan-out / fan-in**：Agent 分支中 analysis 节点同时派发 sql_node（NL2SQL 查案例库）与 research（向量检索法条），synthesis 节点作为 fan-in 屏障等待两者完成后综合生成
- **Human-in-the-Loop 人工介入**：复杂问题生成回答前，把对问题的法律分析（领域/摘要/检索关键词）以确认卡片推给用户确认或修改，用户确认后才继续检索与生成——避免错误检索方向浪费算力
- **流式 SSE 输出**：astream_events 逐 token 推送 + 节点进度提示（"正在检索法条...""正在撰写回答..."）+ 前端打字机效果
- **多轮对话记忆**：MemorySaver checkpointer 按 thread_id（=session_id）持久化 messages，跨轮次承接指代（"他后来怎样了"）
- **结构化法律分析**：每次回答按"行为分析 / 法律后果（分情形）/ 行动建议 / 法律依据 / 免责声明"五段式输出，普通人也能看懂
- **双链路智能路由**
  - **Fast Path**：普通法律咨询 → 单次 LLM 调用直连 RAG，低延迟
  - **Agent 分支**：案例检索/统计类问题 → analysis→{sql_node, research}→synthesis 并行处理
- **条文级 RAG**：按“第 X 条”切分 59 部法律为 7912 个片段，metadata 记录法律名与条文号；BGE-M3 向量化 + Milvus 检索，L2 阈值 0.9、Top-5，低于阈值直接拒答，不做无关硬答
- **引用核验**：答案生成后抽取《法律名》第 X 条，按 metadata 精确回查语料库，未命中引用单独记录；主集引用未落地率由 24.6% 降至 0.8%
- **三层评测体系**：60 题主集 + 50 题 holdout + 25 题对抗集（后两者不参与调参），覆盖路由、检索参数与端到端；指标含路由准确率、条文级 recall@K、引用未落地率、gold 引用召回、P50/P95、LLM 调用数与正确拒答率
- **NL2SQL 案例查询**：自然语言直接查询结构化案例库，自动清洗 LLM 生成 SQL 中的 Markdown 包裹
- **稳定的结构化输出**：PydanticOutputParser 约束 JSON Schema，解析失败自动降级兜底，服务不中断
- **现代化 Web 界面**：气泡式对话框、思考中加载动画、流式打字光标、HITL 确认卡片、多轮会话、Enter 发送 / Shift+Enter 换行

---

## 🧱 技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                    浏览器 (index.html)                        │
│            SSE 流式接收 + HITL 确认卡片交互                    │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP /chat  /resume
┌───────────────────────────▼──────────────────────────────────┐
│                    FastAPI (main.py)                         │
│              lifespan 启动装载 / SSE StreamingResponse        │
└───────────────────────────┬──────────────────────────────────┘
                            │ ask_stream / resume_stream
              ┌─────────────▼────────────────────┐
              │     LangGraph 状态图 (graph.py)  │
              │     MemorySaver checkpointer      │
              │     thread_id = session_id        │
              └─────────────┬────────────────────┘
                            │
                ┌───────────▼───────────┐
                │   supervisor 节点      │  条件路由
                │   needs_sql 判断       │
                └───┬───────────────┬───┘
        普通咨询   │               │  案例/统计查询
        ┌──────────▼──────────┐   ┌──▼────────────────────────┐
        │  fast_rag 节点       │   │  analysis 节点              │
        │  RAG 检索 → LLM(1次) │   │  with_structured_output     │
        └──────────┬──────────┘   │  生成分析草稿               │
                   │              └──┬─────────────────────────┘
                   │            ┌────▼─────────────────────┐
                   │            │ analysis_confirm 节点    │  Human-in-the-Loop
                   │            │ interrupt 暂停等用户确认 │
                   │            │ 用户修改关键词后 resume   │
                   │            └────┬─────────────────────┘
                   │          (fan-out)并行
                   │            ┌────┴────┐
                   │        ┌───▼──┐  ┌────▼───┐
                   │        │sql_  │  │research│
                   │        │node  │  │节点    │
                   │        │NL2SQL│  │向量检索 │
                   │        └───┬──┘  └────┬───┘
                   │          (fan-in)屏障
                   │            ┌────┴────▼┐
                   │            │synthesis  │  综合生成五段式
                   │            │节点       │  astream 流式输出
                   │            └────┬─────┘
                   └────────────────┬▼┐
                          ┌─────────▼─▼─┐
                          │ parse 节点  │  JSON 解析/透传兜底
                          │ fallback    │  失败降级
                          └─────────────┘
                            │
                   ┌────────▼────────┐    ┌───────────────────┐
                   │  Milvus 向量库   │    │  MySQL 法律案例库  │
                   │  BGE-M3(Ollama)  │    │  (NL2SQL)          │
                   └─────────────────┘    └───────────────────┘
```

> 生成结束后，`fast_rag` 与 `synthesis` 都会调用 `citations.verify_and_attach()`：抽取答案中的《法名》第 X 条，按 metadata 回查 Milvus，未命中则记录告警，避免“答案引用了检索上下文里并不存在的法条”。

### 技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | **LangGraph 1.x**（StateGraph / 条件路由 / fan-out 并行 / MemorySaver / Interrupts） |
| 后端 | Python、FastAPI、LangChain、Pydantic v2 |
| 大模型 | DeepSeek-V3（OpenAI 兼容 API） |
| 嵌入模型 | BGE-M3（Ollama 本地部署） |
| 向量数据库 | Milvus 3.x（etcd + MinIO 依赖） |
| 关系数据库 | MySQL 8.0（法律案例库） |
| 前端 | 原生 HTML/CSS/JS（SSE 流式 + 打字机 + HITL 卡片） |
| 基础设施 | Docker Compose |

---

## 📁 项目结构

```
legal-chatbot/
├── app/
│   ├── main.py             # FastAPI 入口、lifespan、/chat /resume SSE 接口
│   ├── agent.py            # ask_stream / resume_stream 流式入口、interrupt 检测
│   ├── graph.py            # LangGraph 图构建(节点+边+条件路由+MemorySaver)
│   ├── state.py            # LegalState TypedDict 共享状态
│   ├── llm.py              # DeepSeek-V3 (ChatOpenAI 兼容) 单例
│   ├── tools.py            # 复用函数:get_db / _get_sql_llm / _extract_sql / similarity_search
│   ├── vector_store.py     # Milvus 初始化、相似度检索、阈值过滤
│   ├── embeddings.py       # Ollama BGE-M3 嵌入封装
│   ├── document_loader.py  # 按“第 X 条”条文级切分，metadata.article
│   ├── citations.py        # 引用抽取、按法律名+条文号回查、核验补全
│   ├── output_parser.py    # 五段式回答 Pydantic Schema
│   ├── config.py           # 全部环境变量配置
│   ├── nodes/              # LangGraph 各节点
│   │   ├── supervisor.py   # 路由:正则判断 needs_sql
│   │   ├── analysis.py     # analysis(结构化分析) + analysis_confirm(interrupt 确认)
│   │   ├── research.py     # 向量检索法条
│   │   ├── sql_node.py     # NL2SQL 查案例库
│   │   ├── synthesis.py    # 综合生成(async + astream 流式)
│   │   ├── fast_rag.py     # Fast Path:检索+1次LLM
│   │   ├── parse.py        # JSON 解析/透传,写回 messages
│   │   └── fallback.py     # 解析失败降级兜底
│   └── static/index.html   # 前端聊天界面(SSE + 打字机 + HITL 卡片)
├── eval/                   # 评测脚本、60 题主集 + 50 题 holdout + 25 题对抗集、评测产物
├── data/documents/         # 59 部现行法律 docx(随仓库分发)
├── init.sql                # MySQL 案例库建表脚本
├── docker-compose.yml      # etcd/minio/milvus/mysql/ollama 编排
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
| `MILVUS_HOST` | **本地运行必填**，填 localhost（默认是容器名 milvus） | `localhost` |
| `MYSQL_HOST` | MySQL 地址 | `localhost` |
| `MYSQL_PASSWORD` | 与 docker-compose 中一致 | `123456` |
| `OLLAMA_BASE_URL` | Ollama 地址（嵌入模型用） | `http://localhost:11434` |
| `RAG_TOP_K` | 检索返回的条文片段数，默认 5 | `5` |
| `RAG_SCORE_THRESHOLD` | L2 距离阈值，越小越严格，默认 0.9 | `0.9` |
| `CITATION_VERIFY` | 设为 `0` 可关闭引用核验（评测对照用） | `1` |

PowerShell 设置示例：

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
$env:MILVUS_HOST="localhost"
```

> PyCharm 用户：在 Run Configuration → Environment variables 中配置上述变量，运行方式选择 **Module name = `uvicorn`**，参数填 `app.main:app --host 0.0.0.0 --port 8000`，工作目录设为项目根目录。**不要使用 reload 模式**（Windows 下子进程会导致监听失败）。

### 5. 启动应用

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 必须用 `workers=1`（默认），多 worker 下 MemorySaver 的线程隔离会失效。

首次启动会自动将 `data/documents/` 下的法律文档分块、向量化并写入 Milvus（约 5-15 分钟，之后启动秒级完成）。

浏览器访问 **http://localhost:8000** 即可使用。

---

## 💡 使用示例

**普通咨询（Fast Path，无需确认）**

> 我在马路上开车撞死了人会被判几年？

回答将按五段式流式输出：行为定性（涉嫌交通肇事罪，关键看责任划分）→ 分情形后果（不构成犯罪 / 3 年以下 / 3-7 年 / 7 年以上 / 民事赔偿）→ 行动建议（停车报警、切勿逃逸、赔偿争取谅解）→ 法律依据（《刑法》第 133 条）→ 免责声明。

**多轮记忆（承接指代）**

> 第一轮：我在马路上开车撞死了人会被判几年？
> 第二轮：那如果他逃逸了呢？

第二轮自动承接上文（"他"=肇事者，"逃逸"=交通肇事罪加重情节），无需重复描述案情。

**案例库查询 + Human-in-the-Loop 确认（Agent 分支）**

> 案例库有多少起交通肇事案件？

问题命中"案例库/多少起"关键词 → 走 Agent 分支 → analysis 节点生成问题分析（领域/摘要/关键词）→ **弹出确认卡片**，展示并可编辑：

- 法律领域：刑事
- 问题摘要：查询交通肇事案件数量
- 检索关键词：交通肇事罪，量刑（可改为更精准的词）

点 **[直接使用]** 或修改后点 **[修改后继续]** → 图恢复执行 → sql_node 查案例库 + research 检索法条并行 → synthesis 综合生成流式回答。

也可以直接调用 API：

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "message": "案例库有多少起交通肇事案件"}'
# 返回 SSE:data: {"type":"interrupt","content":{...分析...},"thread_id":"test-001"}

curl -N -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "resume_value": {"action": "approve"}}'
# 返回 SSE:status 查询/检索/撰写 → token 流式 → done
```

---

## 🔧 关键设计说明

- **为什么从 if-else 路由迁移到 LangGraph 状态图？**
  原架构（`_agent_answer` + `_fast_rag_answer` 两条独立链路 + if-else 路由）存在逻辑重复、扩展困难、无法并行等问题。LangGraph 状态图让节点协作显式化：supervisor 路由 → analysis → {sql, research} 并行 → synthesis 屏障 → parse → END，新增能力只需加节点和边。

- **为什么把 analysis 拆成 analysis + analysis_confirm 两个节点？**
  LangGraph 的 `interrupt` 恢复时会**从头重新执行整个节点**。若 interrupt 与 DeepSeek LLM 调用同处一节点，用户确认后会再调一次 LLM（耗时 + 费钱）。拆分后：analysis 节点跑 LLM 生成结构化草稿（只执行一次），analysis_confirm 节点负责 interrupt 暂停与恢复（廉价，可重跑）。这是 LangGraph 官方推荐的 HITL 写法。

- **fan-out / fan-in 并行模式**
  analysis_confirm 完成后同时派发 sql_node（NL2SQL 查 MySQL 案例库）和 research（向量检索法条），synthesis 节点作为 fan-in 屏障等待两者都完成才综合生成。两个节点写不同 state 字段（case_data / retrieved_docs），无写冲突，无需额外 reducer。

- **多轮记忆：MemorySaver + thread_id**
  checkpointer 按 `thread_id=session_id` 持久化 `messages`（用 `add_messages` reducer 自动追加）。synthesis/fast_rag 节点把最近 `HISTORY_LIMIT` 轮历史拼进提示词，让 LLM 能承接指代。无需手动管理 history 列表。

- **Fast Path + Agent 双链路为什么保留？**
  Agent 分支每次回答需要多节点协作（analysis + sql + research + synthesis）。绝大多数普通咨询只需"检索 + 生成"，Fast Path 把 LLM 调用降为 1 次；只有案例库统计这类必须操作结构化数据的请求才走 Agent 分支并人工确认。

- **Token 预算管控**
  系统提示词（约 1300 tokens）+ 5 条条文片段 + 输出预留（2048）必须小于模型上下文窗口。曾因预算超限导致系统指令被静默截断、模型退化为机械罗列法条，通过 `RAG_TOP_K=5` + 窗口核算修复。

- **条文级切分与 metadata**
  按“第 X 条”把每部法律切成独立片段，metadata 同时写入 `source`（法律名）与 `article`（条文号）；只有超长条文才回退到 1200 字符递归切分。这样检索命中后可直接按条文号回查，而不是依赖文本相似度。

---

## 📊 评测

评测代码在 `eval/`，结果产物在 `eval/out/`。分为三层，互不影响：

```bash
# 主集（60 题，默认读取 eval/questions.jsonl）
python eval/run_eval.py --suite route
python eval/run_eval.py --suite retrieval --ks 3 5 8 --thresholds 0.8 1.0
python eval/run_eval.py --suite e2e

# holdout（50 题，不参与调参；--prefix 避免覆盖主集结果）
python eval/run_eval.py --suite route --questions eval/holdout.jsonl --prefix holdout_
python eval/run_eval.py --suite e2e --questions eval/holdout.jsonl --prefix holdout_

# 对抗集（25 题：多法条竞合、时效、口语化长问题）
python eval/run_eval.py --suite route --questions eval/adversarial.jsonl --prefix adversarial_
python eval/run_eval.py --suite e2e --questions eval/adversarial.jsonl --prefix adversarial_

# 冒烟测试：只跑前 5 题
python eval/run_eval.py --suite e2e --limit 5
```

- `route` 与 `retrieval` 不调用大模型，无需配置 `DEEPSEEK_API_KEY`；`e2e` 需要 Key + Milvus。
- `--questions` 切换题集，`--prefix` 决定输出文件名前缀（如 `holdout_e2e_summary.md`）。
- `--ks` / `--thresholds` 仅对 `retrieval` 生效，不传时用脚本默认值。

### 主集、holdout 与对抗集

| 数据集 | 题数 | 路由准确率 | 要点命中 | 引用未落地率 | gold 引用召回 | 正确拒答 | 平均 LLM 调用 | P50 / P95 |
|---|---|---|---|---|---|---|---|---|
| 主集（调参） | 60 | 100.0% | 97.3% | 0.8% | 100.0% | 100.0% | 1.42 | 4.17s / 5.04s |
| holdout | 50 | 96.0% | 95.0% | 3.2% | 77.1% | 90.0% | 1.30 | 3.74s / 5.32s |
| 对抗集 | 25 | 100.0% | 100.0% | 3.3% | 88.0% | — | 1.00 | 4.23s / 5.00s |

未落地率指答案引用中无法在语料库定位到原文的比例；holdout 与对抗集高于主集，说明换问法后检索漏召回更明显，主集 100% gold 召回是调参后的过拟合信号。

主集平均 LLM 调用 **1.42 次**；普通咨询走 Fast Path 时固定 1 次，只有案例统计类问题进入 Agent 多节点链路。

### 检索参数选型

40 题带法条标注的检索评测中，Top-K 与 L2 阈值做网格对比：

| Top-K | L2 阈值 | 条文级 recall@K | 来源级 recall@K | 来源无关文档占比 |
|---|---|---|---|---|
| 3 | 0.8 | 95.0% | 100.0% | 23.3% |
| 5 | 0.8 | 100.0% | 100.0% | 27.5% |
| 8 | 0.8 | 100.0% | 100.0% | 29.9% |

k=5 已能覆盖全部标注条文，且无关文档少于 k=8，因此固定 `RAG_TOP_K=5`；线上 L2 阈值取 0.9（正常法律问题最小距离 ≤0.81，非法律问题 ≥0.95）。

---

## ⚠️ 免责声明

本项目仅供学习与技术交流，所有 AI 生成内容**不构成法律意见**。真实法律问题请咨询执业律师。

---

## 📄 License

MIT

## 👤 作者

**唐富焱**（[GitHub: tang-fy](https://github.com/tang-fy)）
