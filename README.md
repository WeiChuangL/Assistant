# 智能体助手 — 技术文档

基于 NVIDIA API（OpenAI 兼容）构建的 RAG 智能体，集成 PostgreSQL + pgvector 向量检索、长期记忆与用户画像系统。

---

## 目录

1. [架构总览](#1-架构总览)
2. [项目结构](#2-项目结构)
3. [模块详解](#3-模块详解)
   - [3.1 配置管理 (config.py)](#31-配置管理-configpy)
   - [3.2 LLM 客户端 (llm/)](#32-llm-客户端-llm)
   - [3.3 向量存储 (vector_store/)](#33-向量存储-vector_store)
   - [3.4 知识库引擎 (knowledge/)](#34-知识库引擎-knowledge)
   - [3.5 记忆系统 (memory/)](#35-记忆系统-memory)
   - [3.6 Agent 核心 (agent/)](#36-agent-核心-agent)
   - [3.7 CLI 界面 (cli/)](#37-cli-界面-cli)
4. [数据库设计](#4-数据库设计)
5. [Agent 运行流程](#5-agent-运行流程)
6. [启动指南](#6-启动指南)
7. [命令参考](#7-命令参考)
8. [技术栈](#8-技术栈)

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI 交互层                            │
│              cli/app.py  +  cli/commands.py                  │
│                (rich 终端美化 + / 命令系统)                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                      Agent 核心层                            │
│                   agent/core.py                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  检索编排    │  │  上下文组装   │  │  后处理           │   │
│  │  并行检索    │  │  提示模板     │  │  记忆提取         │   │
│  │  知识库+记忆 │  │  agent/       │  │  画像更新         │   │
│  │  +画像       │  │  prompts.py   │  │                   │   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                      领域服务层                               │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐   │
│  │  knowledge/   │  │   memory/      │  │   llm/        │   │
│  │  - ingestion  │  │  - short_term  │  │  - client     │   │
│  │  - retrieval  │  │  - long_term   │  │  - models     │   │
│  │  - manager    │  │  - profile     │  │               │   │
│  └───────┬───────┘  └───────┬────────┘  └───────┬───────┘   │
└──────────┼──────────────────┼───────────────────┼───────────┘
           │                  │                   │
┌──────────▼──────────────────▼───────────────────▼───────────┐
│                      基础设施层                               │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  vector_store/   │  │   PostgreSQL + pgvector          │ │
│  │  - db.py         │  │   172.16.3.20:5432/assistant     │ │
│  │  - chunker.py    │  │                                  │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  config.py       │  │   NVIDIA API                     │ │
│  │  (pydantic       │  │   integrate.api.nvidia.com/v1    │ │
│  │   settings)      │  │   Chat + Embedding               │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 项目结构

```
Assistant/
├── main.py                      # 启动入口
├── .env                         # 环境变量（API Key、数据库连接等）
├── .env.example                 # 环境变量模板
├── pyproject.toml               # 项目依赖声明
├── uv.lock                      # 依赖锁定文件
├── README.md                    # 本文件
├── demo.py                      # NVIDIA API 原始 demo（保留）
│
└── src/
    ├── config.py                # 配置管理
    │
    ├── llm/                     # LLM 调用层
    │   ├── client.py            # NVIDIA API 客户端
    │   └── models.py            # 模型名称常量
    │
    ├── vector_store/            # 向量存储层
    │   ├── db.py                # PG 连接、建表、类型工具
    │   └── chunker.py           # 文档分块
    │
    ├── knowledge/               # 知识库引擎
    │   ├── ingestion.py         # 文档摄入
    │   ├── retrieval.py         # 语义检索
    │   └── manager.py           # 知识库管理
    │
    ├── memory/                  # 记忆系统
    │   ├── short_term.py        # 短期对话记忆
    │   ├── long_term.py         # 长期向量记忆
    │   └── profile.py           # 用户画像
    │
    ├── agent/                   # Agent 核心
    │   ├── core.py              # Agent 主循环
    │   └── prompts.py           # 系统提示模板
    │
    └── cli/                     # 交互界面
        ├── app.py               # CLI 入口
        └── commands.py          # / 命令处理
```

---

## 3. 模块详解

### 3.1 配置管理 (`config.py`)

基于 `pydantic-settings` 实现类型安全的配置加载，自动从 `.env` 文件读取。

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `nvidia_api_key` | str | NVIDIA API 密钥 | - |
| `nvidia_base_url` | str | API 地址 | `integrate.api.nvidia.com/v1` |
| `llm_chat_model` | str | 对话模型 | `deepseek-ai/deepseek-v4-flash` |
| `llm_embedding_model` | str | Embedding 模型 | `nvidia/nv-embedqa-e5-v5` |
| `pg_*` | - | PostgreSQL 连接参数 | localhost:5432 |
| `agent_short_term_size` | int | 短期记忆窗口大小 | 20 |
| `agent_top_k_chunks` | int | 知识库检索数量 | 5 |
| `agent_top_k_memories` | int | 记忆检索数量 | 3 |
| `agent_similarity_threshold` | float | 相似度阈值 | 0.3 |
| `agent_chunk_size` | int | 分块大小 | 500 |
| `agent_chunk_overlap` | int | 分块重叠 | 50 |

---

### 3.2 LLM 客户端 (`llm/`)

#### `models.py` — 模型常量

定义可用模型名称映射表，避免代码中硬编码。支持通过 `CHAT_MODELS` 和 `EMBEDDING_MODELS` 字典管理，同时维护各 embedding 模型的向量维度。

#### `client.py` — API 封装

全局单例 `llm_client`，提供三个核心能力：

| 方法 | 说明 | 参数 |
|------|------|------|
| `embed(texts)` | 生成查询 embedding | `input_type="query"` |
| `embed_documents(texts)` | 生成文档 embedding | `input_type="passage"` |
| `chat_stream(messages)` | 流式对话 | temperature, top_p, max_tokens |
| `chat(messages)` | 非流式对话 | 同上 |

关键设计：
- `embed` 和 `embed_documents` 使用 NVIDIA embedding 模型的 `input_type` 参数区分 query/passage，获得更好的检索精度
- `chat_stream` 支持 reasoning（思维链）内容的实时输出

---

### 3.3 向量存储 (`vector_store/`)

#### `db.py` — 数据库管理

| 函数 | 说明 |
|------|------|
| `vec_to_str(embedding)` | 将 `list[float]` 转为 pgvector 兼容字符串 `[1.0,2.0,...]` |
| `get_conn()` | 获取异步 psycopg 连接 |
| `init_db()` | 建表 + 创建向量索引（ivfflat） |

> 注意：pgvector 扩展需由超级用户在数据库中手动执行 `CREATE EXTENSION vector;` 安装，应用层不做此操作。

#### `chunker.py` — 文档分块

采用**段落优先 + 句子兜底**的分块策略：

1. 按空行分割段落
2. 累计段落直到接近 `chunk_size`
3. 如单个段落超长，按句子边界二次分割
4. 保留 metadata（来源、索引）

---

### 3.4 知识库引擎 (`knowledge/`)

#### `ingestion.py` — 文档摄入

完整的摄入流水线：

```
文件读取 → 文本提取 → 智能分块 → 批量 Embedding → 写入 pgvector
```

| 函数 | 说明 |
|------|------|
| `read_file(path)` | 自动识别格式（txt/md/pdf），提取文本 |
| `ingest_file(path)` | 摄入单个文件，返回 document_id |
| `ingest_directory(path)` | 递归摄入目录下所有支持的文件 |

支持格式：`.txt`、`.md`、`.pdf`

#### `retrieval.py` — 语义检索

核心函数 `search_chunks(query, top_k, threshold, document_id)`：

1. 将查询文本通过 NVIDIA API 转为 embedding 向量
2. 在 pgvector 中执行余弦相似度检索
3. 使用子查询结构避免 psycopg 参数绑定问题
4. 返回 `RetrievalResult` 列表，包含内容、分数、来源文件

#### `manager.py` — 知识库管理

| 函数 | 说明 |
|------|------|
| `list_documents()` | 列出所有文档（含分块数统计） |
| `delete_document(id)` | 级联删除文档及其所有分块 |
| `get_chunks_for_doc(id)` | 预览指定文档的分块内容 |

---

### 3.5 记忆系统 (`memory/`)

#### `short_term.py` — 短期记忆

基于 `collections.deque` 的滑动窗口实现：

- 自动限制最大条目数（默认 20）
- 提供 `add()`、`get_all()`、`get_last_n()`、`clear()` 操作
- 存储角色（user/assistant）和内容

#### `long_term.py` — 长期记忆

将重要信息向量化存储，支持跨会话回忆：

| 函数 | 说明 |
|------|------|
| `store_memory(content, summary, type, importance)` | 存储一条记忆及其 embedding |
| `search_memories(query, top_k, threshold)` | 语义搜索历史记忆 |
| `extract_and_store_memories(convo)` | LLM 自动从对话中提取并存储记忆 |

记忆类型：`fact`（事实）、`preference`（偏好）、`conversation`（对话）

#### `profile.py` — 用户画像

键值对存储用户偏好信息：

| 函数 | 说明 |
|------|------|
| `get_profile()` | 获取完整画像 |
| `set_profile_value(key, value)` | 设置单条画像 |
| `extract_and_update_profile(convo)` | LLM 自动识别并更新偏好 |

---

### 3.6 Agent 核心 (`agent/`)

#### `core.py` — Agent 主循环

`Agent` 类封装完整的 RAG 对话流程：

```
Step 1: 生成 Query Embedding
Step 2: 并行检索（知识库 + 长期记忆 + 用户画像）
Step 3: 组装系统提示（注入画像、记忆、检索结果、对话历史）
Step 4: LLM 流式生成（用户看到 token-by-token 输出）
Step 5: 后处理（更新短期记忆 → LLM 提取长期记忆 → 更新画像）
```

关键设计：
- **并行检索**：知识库、记忆、画像三个查询并发执行
- **流式输出**：用户无需等待完整响应
- **异步后处理**：记忆提取在后台执行，不阻塞下一次对话

#### `prompts.py` — 系统提示模板

定义了智能体的角色定位和行为规范：
- 优先使用知识库内容回答，标注来源
- 知识库无结果时声明"基于通用知识"
- 引用历史记忆时标注"根据之前的记录"

---

### 3.7 CLI 界面 (`cli/`)

#### `app.py` — 交互入口

基于 `rich` 库的终端交互循环，区分两种输入：

- **以 `/` 开头** → 路由到命令处理器
- **普通文本** → 触发 Agent 对话，流式输出

#### `commands.py` — 命令系统

| 命令 | 功能 | 实现 |
|------|------|------|
| `/kb add <path>` | 导入文档/目录 | → `ingestion.ingest_file/directory` |
| `/kb list` | 文档列表 | → `manager.list_documents` |
| `/kb search <q>` | 搜索知识库 | → `retrieval.search_chunks` |
| `/kb delete <id>` | 删除文档 | → `manager.delete_document` |
| `/memory search <q>` | 搜索记忆 | → `long_term.search_memories` |
| `/profile show` | 查看画像 | → `profile.get_profile` |
| `/profile set <k> <v>` | 设置画像 | → `profile.set_profile_value` |
| `/profile del <k>` | 删除画像 | → `profile.delete_profile_value` |
| `/clear` | 清除对话记忆 | → `agent.clear_memory` |
| `/help` | 帮助 | 打印命令列表 |
| `/exit` | 退出 | SystemExit |

---

## 4. 数据库设计

### ER 关系

```
documents (1) ────< (N) chunks
memories (独立表)
user_profile (独立表)
```

### 表结构

#### `documents` — 文档元数据

| 列 | 类型 | 说明 |
|----|------|------|
| id | SERIAL PK | 自增主键 |
| filename | TEXT | 文件名 |
| file_path | TEXT | 文件路径 |
| file_type | TEXT | 扩展名 |
| created_at | TIMESTAMP | 创建时间 |

#### `chunks` — 文档分块 + 向量

| 列 | 类型 | 说明 |
|----|------|------|
| id | SERIAL PK | 自增主键 |
| document_id | INTEGER FK | 关联 documents.id（级联删除） |
| content | TEXT | 分块文本 |
| chunk_index | INTEGER | 分块序号 |
| embedding | vector(1024) | 文本 embedding 向量 |
| metadata | JSONB | 额外元数据 |

索引：`ivfflat` on `embedding vector_cosine_ops`

#### `memories` — 长期记忆

| 列 | 类型 | 说明 |
|----|------|------|
| id | SERIAL PK | 自增主键 |
| content | TEXT | 记忆内容 |
| summary | TEXT | 简短摘要（用于 embedding） |
| embedding | vector(1024) | 摘要 embedding |
| memory_type | TEXT | fact / preference / conversation |
| importance | FLOAT | 重要度 0-1 |
| metadata | JSONB | 额外元数据 |
| last_accessed | TIMESTAMP | 最后访问时间 |

索引：`ivfflat` on `embedding vector_cosine_ops`

#### `user_profile` — 用户画像

| 列 | 类型 | 说明 |
|----|------|------|
| id | SERIAL PK | 自增主键 |
| key | TEXT UNIQUE | 属性名 |
| value | TEXT | 属性值 |
| updated_at | TIMESTAMP | 更新时间 |

---

## 5. Agent 运行流程

### 对话时序

```
┌────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐
│  User  │     │   CLI    │     │  Agent   │     │  NVIDIA  │     │  PG    │
└───┬────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     └───┬────┘
    │  输入 Query   │                │                │               │
    │──────────────>│                │                │               │
    │               │  chat_stream() │                │               │
    │               │───────────────>│                │               │
    │               │                │  embed(query)  │               │
    │               │                │───────────────>│               │
    │               │                │<─── vec[1024] ─│               │
    │               │                │                │               │
    │               │                │─── 并行检索 ────┼──────────────>│
    │               │                │  知识库 + 记忆 + 画像           │
    │               │                │<── 结果集 ─────┼───────────────│
    │               │                │                │               │
    │               │                │  组装上下文    │               │
    │               │                │  chat_stream() │               │
    │               │                │───────────────>│               │
    │               │                │<── token流 ────│               │
    │               │<─── yield ─────│                │               │
    │<── 逐字显示 ──│                │                │               │
    │               │                │                │               │
    │               │                │  后处理:       │               │
    │               │                │  提取记忆+画像 │               │
    │               │                │───────────────>│               │
    │               │                │───────────────┼──────────────>│
```

### 上下文组装顺序

最终发送给 LLM 的 messages 结构：

```
[0] system prompt
    ├── 角色设定（智能助手，基于知识库回答）
    ├── 用户画像（偏好、背景）
    ├── 长期记忆（相关历史信息，按重要度排序）
    ├── 知识库检索结果（文档片段 + 来源引用 + 相关度分数）
    └── 最近对话历史（最近6轮）

[1] user message 1
[2] assistant message 1
...
[N] 当前用户输入
```

---

## 6. 启动指南

### 前置条件

1. **Python 3.13+**
2. **PostgreSQL + pgvector 扩展已安装**
3. **NVIDIA API Key**（从 build.nvidia.com 获取）

### 快速启动

```bash
# 1. 克隆 / 进入项目目录
cd Assistant

# 2. 安装依赖
uv sync

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 NVIDIA API Key 和 PostgreSQL 连接信息

# 4. 确保 pgvector 扩展已安装
# 在 PostgreSQL 中执行（需要超级用户）：
# CREATE EXTENSION vector;

# 5. 启动
python main.py
```

### 启动输出示例

```
Initializing database...
Database ready.

智能助手已启动
输入 /help 查看命令，/exit 退出

>
```

### 导入知识库示例

```
> /kb add ./docs/
  正在导入: ./docs/
  [OK] ./docs/api-guide.md
  [OK] ./docs/architecture.pdf
  导入完成，共 2 个文件

> 我们系统的认证方式是什么？
  [检索知识库...]
  根据知识库中的架构文档，系统使用 JWT Token 认证...
  [来源: architecture.pdf]
```

---

## 7. 命令参考

### 知识库管理 `/kb`

| 子命令 | 用法 | 说明 |
|--------|------|------|
| `add` | `/kb add <文件/目录>` | 导入文档，自动分块+向量化 |
| `list` | `/kb list` | 列出所有文档，含分块数和导入时间 |
| `search` | `/kb search <查询>` | 直接搜索知识库（不经 LLM） |
| `delete` | `/kb delete <ID>` | 删除文档及其所有分块 |

### 记忆管理 `/memory`

| 子命令 | 用法 | 说明 |
|--------|------|------|
| `search` | `/memory search <查询>` | 搜索长期记忆 |

### 用户画像 `/profile`

| 子命令 | 用法 | 说明 |
|--------|------|------|
| `show` | `/profile show` | 查看当前用户画像 |
| `set` | `/profile set <键> <值>` | 手动设置画像 |
| `del` | `/profile del <键>` | 删除画像条目 |

### 其他

| 命令 | 说明 |
|------|------|
| `/clear` | 清除当前对话记忆（短期记忆） |
| `/help` | 显示帮助信息 |
| `/exit` | 退出程序 |

---

## 8. 技术栈

| 层次 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 运行时 | Python | 3.13+ | 异步原生支持 |
| LLM 网关 | NVIDIA API | - | OpenAI 兼容，免费额度 |
| 对话模型 | DeepSeek V4 Flash | - | 主力推理（可切换） |
| Embedding | NV-EmbedQA-E5-V5 | - | 1024 维向量 |
| 向量库 | PostgreSQL + pgvector | PG 17 + v0.7.4 | 本地部署 |
| DB 驱动 | psycopg | 3.x (binary) | 异步连接 |
| 配置 | pydantic-settings | 2.x | 类型安全配置 |
| CLI | rich | 13.x | 终端美化 |
| PDF | pypdf | 5.x | PDF 解析 |
| Markdown | markdown-it-py | 3.x | MD 解析 |
| 包管理 | uv | - | 快速依赖管理 |

### 模型切换

对话模型可通过 `.env` 中的 `LLM_CHAT_MODEL` 切换，可选：

| 模型标识 | 模型全名 |
|----------|----------|
| `deepseek-v4-flash` | `deepseek-ai/deepseek-v4-flash` |
| `deepseek-v4-pro` | `deepseek-ai/deepseek-v4-pro` |
| `glm-5.1` | `z-ai/glm-5.1` |
| `qwen3-coder` | `qwen/qwen3-coder-480b-a35b-instruct` |
| `minimax-m2.7` | `minimaxai/minimax-m2.7` |
| `gemma-4-31b` | `google/gemma-4-31b-it` |

### 未来扩展路线

- [ ] Function Calling 工具系统
- [ ] ReAct 自主推理循环
- [ ] FastAPI Web API
- [ ] Web 聊天界面
- [ ] 混合搜索（语义 + BM25）
- [ ] Cross-encoder 重排序
- [ ] 多知识库隔离
- [ ] 会话持久化与历史管理
