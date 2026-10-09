# Deep Agents 自部署方案

本文说明如何把 `examples/` 下基于 Deep Agents 的 agent（以 `deep_research` 为例）部署到生产环境，全程只使用开源组件，不依赖需要 license 的 LangSmith Deployment / Agent Server。

## 1. 方案概览

`create_deep_agent` 返回的是一个编译好的 LangGraph 图，本身只是一个 Python 对象。生产部署要补齐三件事：

1. **持久化**：会话状态（checkpoint）写入自己的 PostgreSQL。
2. **服务层**：用 FastAPI 暴露 HTTP 接口，通过 AG-UI 协议以 SSE 流式输出事件。
3. **前端**：任意 AG-UI 客户端（CopilotKit 或自研界面）连接服务层。

```
┌──────────────┐   AG-UI (HTTP POST + SSE)   ┌─────────────────────────────┐
│   前端 UI     │ ─────────────────────────▶ │  FastAPI 服务 (uvicorn)      │
│ CopilotKit / │ ◀───────────────────────── │  ├─ ag-ui-langgraph 端点     │
│ 自研客户端    │        事件流               │  └─ deep agent (LangGraph)   │
└──────────────┘                             └──────┬───────────────┬──────┘
                                                    │               │
                                          checkpoint│               │模型 / 工具调用
                                                    ▼               ▼
                                             ┌────────────┐   ┌──────────────────┐
                                             │ PostgreSQL │   │ LLM API / Tavily │
                                             └────────────┘   └──────────────────┘
```

### 组件与许可证

| 组件 | 作用 | 许可证 |
|---|---|---|
| `deepagents`、`langchain`、`langgraph` | agent 逻辑与图执行 | MIT |
| `langgraph-checkpoint-postgres` | 把 checkpoint 存到 PostgreSQL | MIT |
| `ag-ui-langgraph` | 把 LangGraph 图适配为 AG-UI 事件流，并挂载到 FastAPI | MIT |
| FastAPI + uvicorn | HTTP 服务 | MIT / BSD |
| PostgreSQL | 状态存储 | PostgreSQL License |

以下组件需要 license 或付费套餐，本方案不使用：`langgraph up` / `langgraph build` 产出的 Agent Server 镜像、LangSmith Deployment（Cloud、Hybrid、Self-hosted）。`langgraph dev` 仅用于本地开发。

## 2. 环境划分

| 阶段 | 运行方式 | 说明 |
|---|---|---|
| 本地开发 | `langgraph dev` + Studio | 内存运行时，热重载，可视化调试图和状态 |
| 本地联调 | `uvicorn server:app --reload` + 本地 PostgreSQL | 与生产同一套服务代码 |
| 生产 | Docker 镜像 + 托管 PostgreSQL | 见第 6 节 |

`langgraph.json` 和 `agent.py` 保持不变，继续服务本地开发；生产入口是新增的 `server.py`。

## 3. 后端实现

### 3.1 依赖

在 agent 项目目录（如 `deep_research/`）中添加：

```bash
uv add fastapi "uvicorn[standard]" ag-ui-langgraph "langgraph-checkpoint-postgres" "psycopg[binary,pool]"
```

### 3.2 拆分 agent 构建逻辑

`agent.py` 在模块级直接创建 `agent`，供 `langgraph dev` 加载。生产服务需要给同一个 agent 注入 checkpointer，因此把构建过程提取为函数，两边共用：

```python
# agent.py
from deepagents import create_deep_agent
from langgraph.types import Checkpointer


def build_agent(checkpointer: Checkpointer = None):
    """创建 deep research agent；checkpointer 由部署环境决定。"""
    return create_deep_agent(
        model=model,
        tools=[tavily_search, think_tool],
        system_prompt=INSTRUCTIONS,
        subagents=[research_sub_agent],
        checkpointer=checkpointer,
    )


# langgraph dev 加载的图，持久化由开发服务器接管
agent = build_agent()
```

### 3.3 服务入口 `server.py`

```python
# server.py
import os
from contextlib import asynccontextmanager

from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent import build_agent

# 连接池先不打开，等应用启动时在 lifespan 里打开
pool = AsyncConnectionPool(
    conninfo=os.environ["DATABASE_URL"],
    max_size=20,
    open=False,
    # AsyncPostgresSaver 要求 autocommit 和 dict_row
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
checkpointer = AsyncPostgresSaver(pool)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()
    # 首次启动时建表，已存在时为空操作
    await checkpointer.setup()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)

add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAgent(
        name="research",
        description="Deep research agent",
        graph=build_agent(checkpointer),
    ),
    path="/agent",
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
```

本地运行：

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agents
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3.4 请求模型

AG-UI 端点接收 `RunAgentInput`，关键字段：

| 字段 | 含义 |
|---|---|
| `threadId` | 会话 ID，映射为 LangGraph 的 `thread_id`，同一 ID 的多轮对话共享 checkpoint |
| `runId` | 本次执行 ID |
| `messages` | 消息列表 |
| `state` | 前端同步给 agent 的状态 |
| `resume` | 从 human-in-the-loop 中断处恢复时携带的决策 |

响应是 SSE 事件流，主要事件：

| agent 行为 | AG-UI 事件 |
|---|---|
| 运行生命周期 | `RUN_STARTED`、`RUN_FINISHED`、`RUN_ERROR` |
| 助手文本 | `TEXT_MESSAGE_START`、`TEXT_MESSAGE_CONTENT`、`TEXT_MESSAGE_END` |
| 工具调用 | `TOOL_CALL_START`、`TOOL_CALL_ARGS`、`TOOL_CALL_END`、`TOOL_CALL_RESULT` |
| 推理过程 | `REASONING_START`、`REASONING_MESSAGE_CONTENT`、`REASONING_END` |
| 状态（todos、files、子 agent） | `STATE_SNAPSHOT`、`STATE_DELTA` |
| 历史消息 | `MESSAGES_SNAPSHOT` |

完整事件定义见 [AG-UI events](https://docs.ag-ui.com/concepts/events)。

### 3.5 调优 `LangGraphAgent`

- `emit_raw_events=False`：不在每个事件里附带原始 LangGraph 事件。deep agent 的状态较大（files、todos），关闭后能明显减小流量。
- `emit_interrupt_outcome=True`：中断时在 `RUN_FINISHED` 中返回结构化的 interrupt 结果，前端用 `resume` 字段恢复。
- `subagent_visibility`：控制子 agent 事件在流中的呈现方式，可选 `SUBAGENT_VISIBILITY_INLINE`、`SUBAGENT_VISIBILITY_ATTRIBUTED`、`SUBAGENT_VISIBILITY_HIDDEN`。

## 4. 前端

任何 AG-UI 客户端都能连接 `POST /agent`：

| 方案 | 适用场景 |
|---|---|
| [CopilotKit](https://docs.copilotkit.ai/langgraph/deep-agents) | 需要现成的 React 聊天组件、前端工具调用、生成式 UI |
| [AG-UI 客户端 SDK](https://docs.ag-ui.com/quickstart/clients) | 自研界面，直接消费事件流 |
| [其他 AG-UI 集成](https://docs.ag-ui.com/integrations) | 终端、移动端等客户端 |

LangChain 的 `useStream`（`@langchain/react`）和 Agent Chat UI 连接的是 Agent Server API，不适用于本方案。

## 5. 可观测性

| 方案 | 做法 |
|---|---|
| LangSmith 追踪 | 设置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY`。追踪独立于部署产品，有免费额度，按 trace 量计费 |
| Langfuse（可自部署） | 通过 LangChain callback 接入，把 handler 放进 `LangGraphAgent` 的 `config={"callbacks": [...]}` |
| 服务日志与指标 | uvicorn 访问日志 + 应用日志；按需接入 OpenTelemetry / Prometheus |

## 6. 容器化部署

### 6.1 Dockerfile

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/agents
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: agents
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 10

volumes:
  pgdata:
```

生产环境中 PostgreSQL 建议使用云厂商托管实例，`app` 部署到 K8s、ECS、Cloud Run（设置最小实例数，避免缩容到零时中断进行中的运行）等平台。

### 6.3 反向代理

一次深度研究可能持续数分钟，SSE 连接需要长时间保持：

```nginx
location /agent {
    proxy_pass http://app:8000;
    proxy_http_version 1.1;
    proxy_buffering off;         # SSE 需要实时透传
    proxy_read_timeout 900s;     # 覆盖最长一次运行
}
```

## 7. 生产清单

- **鉴权**：AG-UI 端点本身不带鉴权。用 FastAPI 依赖或网关校验用户身份，例如 `add_langgraph_fastapi_endpoint(..., dependencies=[Depends(verify_user)])`。
- **会话归属**：`threadId` 由前端传入，服务端要校验它属于当前用户，防止读取他人会话。可以把 `thread_id → user_id` 的映射存在同一个 PostgreSQL 中。
- **密钥管理**：模型和 Tavily 的 API key 通过密钥管理服务注入，不打进镜像。
- **限流与成本**：每次运行会调用多次模型和搜索。按用户限制并发和频率，并在 `RunnableConfig` 中设置 `recursion_limit`，防止失控循环。
- **数据保留**：checkpoint 会持续增长，定期清理过期 thread（`AsyncPostgresSaver.adelete_thread`）。
- **多实例**：服务本身无状态，状态都在 PostgreSQL 中，可以水平扩容。同一个 thread 的并发运行需要在业务层加锁或拒绝。
- **超长任务**：如果运行时长超出 HTTP 连接能承受的范围，改为「提交任务 → 后台 worker（arq、Celery 等）执行 → 前端按 `threadId` 轮询状态或重新订阅」的模式。checkpoint 按步骤保存，worker 中断后可以从最近一步继续。

## 8. 与 Agent Server 的能力对照

| 能力 | Agent Server | 本方案 |
|---|---|---|
| 流式输出 | 内置 | AG-UI SSE |
| 会话持久化 | 内置 | `AsyncPostgresSaver` |
| human-in-the-loop 中断恢复 | 内置 API | AG-UI `resume` 字段 |
| 后台任务队列、重试 | 内置 | 需自建（第 7 节） |
| assistants / threads 管理 API | 内置 | 按需自行实现 |
| 定时任务（cron） | 内置 | 用外部调度器触发 |
| 鉴权、多租户 | 自定义 auth 插件 | FastAPI 依赖 / 网关 |
| Studio 调试 | 支持 | 开发阶段用 `langgraph dev` |
| 费用 | 生产需 license | 仅基础设施与模型费用 |

## 参考

- [Deep Agents AG-UI](https://docs.langchain.com/oss/python/deepagents/ag-ui)
- [CopilotKit 集成（含 FastAPI 桥接）](https://docs.langchain.com/oss/python/langchain/frontend/integrations/copilotkit)
- [LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)
- [ag-ui-langgraph (PyPI)](https://pypi.org/project/ag-ui-langgraph/)
- [AG-UI 协议](https://docs.ag-ui.com)
