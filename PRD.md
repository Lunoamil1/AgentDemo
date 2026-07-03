# PRD 产品需求文档 — 智能扫地机器人智能客服系统（AgentDemo）

## 1. 产品概述

### 1.1 产品名称
智能扫地机器人智能客服系统（AgentDemo）

### 1.2 产品定位
基于 ReAct（Reasoning + Acting）智能体架构的扫地机器人 / 扫拖一体机器人垂直领域智能客服系统。系统融合 RAG（检索增强生成）向量知识库检索与多工具协同调用能力，为用户提供涵盖产品选购、功能答疑、故障排查、保养维护、使用教程全场景的一站式专业咨询服务。

### 1.3 目标用户
| 用户类型 | 场景描述 |
|---------|---------|
| 售前咨询用户 | 需要选购扫地机器人 / 扫拖一体机器人，对参数、机型、适用场景不了解 |
| 售后使用用户 | 已购机用户，遇到使用问题、故障报错、保养维护等需求 |
| 场景化用户 | 有特定场景需求（养宠、母婴、木地板、大户型、潮湿地区等），需定制化建议 |

### 1.4 核心价值
- **专业精准**：所有回答严格基于官方知识库与工具返回结果，杜绝主观臆断与错误指导
- **自主推理**：ReAct 架构使 Agent 能自主思考、判断工具调用、多轮推理，无需人工编排
- **全场景覆盖**：覆盖售前选购、售中使用、售后维护全生命周期

---

## 2. 功能架构

### 2.1 系统架构总览

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit Web UI                  │
│              (用户交互 / 流式对话展示)                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  ReactAgent (核心引擎)                 │
│         ReAct 闭环: 思考→行动→观察→再思考→回答          │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ 系统提示词  │  │  中间件(Middleware) │  │  Chat Model   │  │
│  │main_prompt│  │ monitor_tools │  │  (Qwen3-Max)  │  │
│  │           │  │ log_before_   │  │               │  │
│  │           │  │ model         │  │               │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ 工具调用
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│rag_summarize│ │ get_weather │ │  get_city   │
│ (RAG检索)    │ │ (天气查询)   │ │ (随机城市)   │
└──────┬──────┘ └─────────────┘ └─────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│              RagSummarizeService (RAG服务)            │
│  ┌────────────────┐  ┌────────────────────────────┐  │
│  │ PromptTemplate  │  │    Retriever (向量检索)      │  │
│  │rag_summarize_  │  │                            │  │
│  │prompt          │  │                            │  │
│  └────────────────┘  └────────────┬───────────────┘  │
└─────────────────────────────────────┼────────────────┘
                                      │
┌─────────────────────────────────────▼────────────────┐
│           VectorStoreService (向量存储服务)             │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ FAISS 向量库  │  │ 文档加载器   │  │ MD5 去重    │  │
│  │ (本地持久化)  │  │ Text/PDF    │  │             │  │
│  └──────────────┘  └─────────────┘  └─────────────┘  │
│  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │ DashScope    │  │ RecursiveCharacterTextSplitter│  │
│  │ Embeddings   │  │ (文档分块)                    │  │
│  └──────────────┘  └─────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 2.2 功能模块清单

| 模块 | 子模块 | 功能说明 |
|------|-------|---------|
| **Web 交互层** | Streamlit UI | 对话式交互界面，流式输出助手回复 |
| **Agent 引擎** | ReactAgent | ReAct 推理引擎，自主调度工具 |
| | 系统提示词 | 定义 Agent 角色与行为规范 |
| | 中间件 | 工具调用监控、模型调用前日志 |
| **工具层** | rag_summarize | RAG 向量知识库检索工具 |
| | get_weather | 城市天气查询工具（模拟） |
| | get_city | 随机城市获取工具 |
| **RAG 服务** | RagSummarizeService | Prompt + 检索 + LLM 组链式调用 |
| **向量存储** | VectorStoreService | 文档加载、去重、分块、向量化、检索 |
| **模型层** | ChatModelFactory | ChatTongyi（Qwen3-Max）聊天模型 |
| | EmbeddingsFactory | DashScope Embeddings（text-embedding-v4）嵌入模型 |
| **基础设施** | 配置管理 | YAML 配置加载（rag / faiss / prompts / agent） |
| | 日志系统 | 控制台 + 文件双输出日志 |
| | 路径工具 | 项目根目录统一绝对路径管理 |
| | 提示词加载 | 系统提示词 / RAG 提示词 / 报告提示词动态加载 |

---

## 3. 功能需求详细说明

### 3.1 对话交互（P0 — 核心）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-001 | 对话式交互 | 用户通过 Streamlit Web 界面与智能客服对话 | 界面显示对话输入框与消息列表 |
| F-002 | 流式输出 | Agent 回复以流式方式逐步展示 | 用户可实时看到回复逐字生成 |
| F-003 | 会话状态保持 | 对话上下文在会话内持续保持 | 同一会话内历史消息持续渲染 |

### 3.2 ReAct 智能推理（P0 — 核心）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-004 | 闭环推理 | 所有用户问题必须走完「思考→行动→观察→再思考→回答」闭环 | Agent 不会跳过思考直接作答 |
| F-005 | 自主工具调度 | Agent 根据问题意图自主判断是否调用工具及调用哪个 | 无需用户显式指定工具 |
| F-006 | 多轮工具联动 | 复杂问题可多轮调用不同工具组合 | 如先调 get_city 再调 get_weather 再调 rag_summarize |
| F-007 | 容错修正 | 工具返回不全时二次调用补全，无匹配数据时如实告知 | 不编造内容、不敷衍作答 |

### 3.3 RAG 知识库检索（P0 — 核心）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-008 | 向量知识库检索 | 通过 FAISS 向量库语义检索扫地机器人专业知识 | 返回与 query 语义相关的文档片段 |
| F-009 | 文档加载 | 支持 .txt 和 .pdf 格式文件加载入库 | 配置中 allow_knowledge_file 控制允许类型 |
| F-010 | 文档分块 | 使用 RecursiveCharacterTextSplitter 按配置分块 | chunk_size=200, chunk_overlap=50 |
| F-011 | MD5 去重 | 基于文件 MD5 校验和去重，已入库文件不重复加载 | md5.txt 记录已入库文件 MD5 |
| F-012 | 向量库持久化 | FAISS 索引本地持久化，启动时自动加载已有索引 | faiss_db 目录存储索引文件 |
| F-013 | 增量入库 | 新文件增量添加到已有向量库，不需全量重建 | 仅新增文件执行分块+向量化 |

### 3.4 工具能力（P1 — 重要）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-014 | RAG 知识检索工具 | rag_summarize：接收 query，返回知识库检索总结 | 所有机器人业务咨询优先调用 |
| F-015 | 天气查询工具 | get_weather：接收城市名，返回天气信息（模拟） | 传入城市名字符串，返回天气文本 |
| F-016 | 随机城市工具 | get_city：无需入参，随机返回城市名称 | 返回北京/上海/广州/深圳/成都之一 |

### 3.5 中间件监控（P2 — 辅助）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-017 | 工具调用监控 | monitor_tools：记录工具名、参数、执行结果 | 日志中可见工具调用全链路 |
| F-018 | 模型调用前日志 | log_before_model：记录当前消息数与最新消息类型 | 日志中可见模型调用前状态 |

### 3.6 日志与配置（P2 — 辅助）

| 需求ID | 需求名称 | 描述 | 验收标准 |
|--------|---------|------|---------|
| F-019 | 双通道日志 | 控制台 INFO + 文件 DEBUG 双通道输出 | logs/ 目录按日期生成日志文件 |
| F-020 | YAML 配置管理 | 所有配置通过 YAML 文件管理，启动时加载 | rag.yml / prompts.yml / agent.yml / faiss.yml |

---

## 4. 数据与知识库

### 4.1 知识库数据文件

| 文件 | 内容概述 |
|------|---------|
| data/扫地机器人100问1.txt | 扫地机器人基础认知、选购参数等 100 问 |
| data/扫地机器人100问2.txt | 扫地机器人进阶问答 |
| data/扫拖一体机器人100问.txt | 扫拖一体机器人专项问答 |
| data/选购指南.txt | 产品选购指南 |
| data/故障排除.txt | 故障排查指南 |
| data/维护保养.txt | 维护保养指南 |

### 4.2 向量库配置

| 配置项 | 值 | 说明 |
|-------|---|------|
| embedding 模型 | text-embedding-v4 | DashScope 文本嵌入模型 |
| chat 模型 | qwen3-max | 通义千问聊天模型 |
| chunk_size | 200 | 文档分块大小（字符数） |
| chunk_overlap | 50 | 分块重叠字符数 |
| persist_directory | faiss_db | FAISS 索引持久化目录 |
| data_path | data | 知识库源文件目录 |
| 允许文件类型 | txt, pdf | 知识库允许入库的文件格式 |

---

## 5. 非功能需求

| 类别 | 需求 | 说明 |
|------|------|------|
| 安全性 | 禁止超范围作答 | Agent 严格遵循工具能力边界，禁止指导拆机维修、硬件改装等 |
| 安全性 | 事实准确 | 所有回答必须基于知识库 / 工具返回结果，禁止编造 |
| 可观测性 | 全链路日志 | 工具调用、模型调用均有日志记录 |
| 可维护性 | 配置外置 | 模型名称、分块参数、路径等全部 YAML 配置化 |
| 可维护性 | 路径统一 | 所有文件路径通过 get_abs_path 获取绝对路径 |
| 可扩展性 | 工具可扩展 | Agent 工具列表可配置，新增工具仅需注册 @tool 装饰器 |
| 可扩展性 | 模型可替换 | 通过 Factory 模式，模型可替换为其他 LangChain 兼容模型 |
| 可扩展性 | 提示词可配置 | 系统提示词、RAG 提示词、报告提示词均外置为独立文件 |

---

## 6. 技术栈

| 层次 | 技术 | 版本/说明 |
|------|------|----------|
| 前端 UI | Streamlit | 1.58.0 |
| Agent 框架 | LangChain + LangGraph | ReAct Agent |
| LLM | 通义千问 Qwen3-Max | 通过 DashScope API |
| Embedding | DashScope text-embedding-v4 | 通过 DashScope API |
| 向量存储 | FAISS | langchain_community.vectorstores.FAISS |
| 文档加载 | PyPDFLoader / TextLoader | langchain_community |
| 文档分块 | RecursiveCharacterTextSplitter | langchain_text_splitters |
| 配置管理 | PyYAML | YAML 配置文件 |
| 日志 | Python logging | 标准库 |
| 语言 | Python | 3.x |

---

## 7. 项目目录结构

```
AgentDemo/
├── app.py                    # Streamlit 应用入口
├── agent/
│   └── react_agent.py        # ReAct Agent 核心引擎
├── tools/
│   ├── agent_tools.py        # Agent 工具定义（rag_summarize / get_weather / get_city）
│   └── middleware.py         # 中间件（工具监控 / 模型调用前日志）
├── rag/
│   ├── vector_store.py       # FAISS 向量存储服务（文档加载/去重/分块/入库/检索）
│   └── rag_service.py        # RAG 检索总结服务（Prompt + 检索 + LLM 链式调用）
├── model/
│   └── factory.py            # 模型工厂（ChatModel / Embeddings）
├── utils/
│   ├── config_handler.py     # YAML 配置加载器
│   ├── logger_handler.py     # 日志管理器
│   ├── path_tool.py          # 项目绝对路径工具
│   ├── file_handler.py       # 文件处理工具（MD5 / 文件筛选 / 文档加载）
│   └── prompt_loader.py      # 提示词加载器
├── config/
│   ├── rag.yml               # RAG 配置（模型 / 分块 / 路径）
│   ├── faiss.yml             # FAISS 配置（预留）
│   ├── prompts.yml           # 提示词路径配置
│   └── agent.yml             # Agent 配置（预留）
├── prompts/
│   ├── main_prompt.txt       # Agent 系统提示词
│   ├── rag_summarize_prompt.txt  # RAG 总结提示词
│   └── report_prompt.txt     # 报告生成提示词
├── data/                     # 知识库源文件目录
│   ├── 扫地机器人100问1.txt
│   ├── 扫地机器人100问2.txt
│   ├── 扫拖一体机器人100问.txt
│   ├── 选购指南.txt
│   ├── 故障排除.txt
│   └── 维护保养.txt
├── faiss_db/                 # FAISS 向量库持久化目录
├── logs/                     # 日志文件目录
└── md5.txt                   # 已入库文件 MD5 去重记录
```

---

## 8. 核心业务流程

### 8.1 用户对话流程

```
用户输入问题
    ↓
Streamlit UI 捕获 prompt
    ↓
ReactAgent.execute_stream(query)
    ↓
Agent 内部 ReAct 循环：
    ┌→ 思考(Thought): 识别意图 → 判断工具需求
    │       ↓
    │  行动(Action): 调用工具（rag_summarize / get_weather / get_city）
    │       ↓
    │  观察(Observation): 接收工具返回结果
    │       ↓
    └─ 再思考: 是否需要补充调用？ ──是──→ 回到行动
           │
           否
           ↓
      最终回答(Final Answer)
           ↓
流式输出至 Streamlit UI
```

### 8.2 RAG 知识库入库流程

```
启动 VectorStoreService
    ↓
检查 FAISS 索引是否已存在
    ├─ 是 → 加载已有向量库
    └─ 否 → 创建空向量库并持久化
    ↓
load_document() 遍历 data/ 目录
    ↓
对每个文件：
    1. 计算文件 MD5
    2. 查询 md5.txt 是否已存在
       ├─ 已存在 → 跳过（去重）
       └─ 不存在 → 继续
    3. 根据扩展名选择 Loader（.txt → TextLoader, .pdf → PyPDFLoader）
    4. 加载文档内容
    5. 记录 MD5 到 md5.txt
    ↓
汇总所有新文档 → RecursiveCharacterTextSplitter 分块
    ↓
FAISS.add_documents() 向量化入库
    ↓
FAISS.save_local() 持久化
```

### 8.3 RAG 检索总结流程

```
用户 query 传入 rag_summarize 工具
    ↓
RagSummarizeService.rag_summarize(query)
    ↓
retriever.invoke(query) → FAISS 相似度检索
    ↓
组装 context（参考资料 + 元数据）
    ↓
PromptTemplate + context + query → LLM
    ↓
StrOutputParser → 返回总结文本
```

---

## 9. 已知限制与后续规划

| 类别 | 当前状态 | 后续规划 |
|------|---------|---------|
| 天气查询 | 模拟数据（固定返回"晴朗,26度"） | 接入真实天气 API |
| 城市获取 | 随机5城列表 | 接入 IP 定位 / 用户信息 |
| 报告生成 | 提示词已准备，工具未接入 | 实现 get_user_id / get_current_month / fetch_external_data 工具 |
| Agent 配置 | agent.yml 为空 | 将 Agent 参数（工具列表、中间件等）配置化 |
| FAISS 配置 | faiss.yml 为空 | 将 FAISS 索引参数配置化 |
| 文件格式 | 仅支持 .txt / .pdf | 扩展支持 Word / Markdown 等 |
| 会话管理 | 单会话无持久化 | 多轮会话持久化与历史回溯 |
