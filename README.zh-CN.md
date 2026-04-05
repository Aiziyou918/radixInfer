# radixInfer

[English](README.md) | [简体中文](README.zh-CN.md)

`radixInfer` 是一个分层清晰的 LLM Serving 系统，具备可运行的端到端控制平面。它将 API 处理、传输层、运行时调度、缓存管理和执行引擎明确拆分。

## 功能特性

- 基于 FastAPI 的 HTTP 服务，支持 SSE 流式输出
- OpenAI 兼容的 `/v1/completions` 和 `/v1/chat/completions` 接口
- 用于本地调试的交互式 Shell 模式
- 多进程架构：API、分词器、运行时分别运行在独立进程中
- Decode-first 重叠调度器（prefill / decode / mixed batch）
- 基于堆的 LRU 淘汰的 Radix 前缀缓存
- 支持前缀共享的分页 KV 缓存
- Llama 系模型工厂（Llama、Mistral、Qwen2、Qwen3、Qwen3-MoE）
- 可插拔的 Attention 后端（FlashInfer / FlashAttention / HF fallback）
- 张量并行支持

## 仓库结构

主代码位于 `python/radixinfer` 下：

| 包 | 职责 |
|---|---|
| `api` | HTTP 路由、请求 schema、SSE/JSON 响应渲染 |
| `server` | 前端编排、监听队列生命周期、后端进程启动 |
| `transport` | ZMQ/队列适配器、协议类型、分词 worker、反分词流程 |
| `runtime` | 调度器、prefill/decode 管理器、缓存管理器、运行时 I/O |
| `cache` | 页池、Radix 前缀缓存、KV 池 |
| `engine` | 模型执行、CUDA Graph、采样、分布式初始化 |
| `models` | 模型注册、配置、权重加载、模型实现 |
| `distributed` | 张量并行通信辅助模块 |
| `layers` | Transformer 基础层（Attention、Linear、Norm、Embedding） |

## 快速开始

### 环境要求

Python 3.10+。核心依赖：`fastapi`、`pyzmq`、`torch`、`transformers>=4.40`、`uvicorn`、`pydantic>=2.0`。

### 启动服务

```bash
# Debug 模型（无需 GPU，无需下载模型）
PYTHONPATH=python python -m radixinfer --model debug --device cpu --port 1919

# 真实模型（需 GPU）
PYTHONPATH=python python -m radixinfer --model Qwen/Qwen3-0.6B --device cuda:0
```

### Shell 模式

```bash
PYTHONPATH=python python -m radixinfer --model debug --device cpu --shell
```

内置命令：`/reset`（清空历史）、`/exit`（退出）。

### 运行测试

```bash
pytest -q
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | HTTP 绑定地址 |
| `--port` | `1919` | HTTP 端口 |
| `--model` | `debug` | 模型名称或路径 |
| `--device` | `auto` | `cpu`、`auto` 或 `cuda:N` |
| `--tp-size` | `1` | 张量并行规模 |
| `--num-pages` | 自动推断 | KV 缓存总页数 |
| `--page-size` | `16` | 每页 token 数 |
| `--max-prefill-length` | `2048` | 每个调度 tick 的 prefill token 预算 |
| `--dist-port` | 自动 | 分布式 rendezvous 端口 |
| `--disable-zmq` | 关闭 | 改用 `mp.Queue` 代替 ZMQ |
| `--shell` | 关闭 | 启动 Shell 模式 |

## API 接口

| 接口 | 说明 |
|---|---|
| `GET /v1/models` | 模型列表 |
| `POST /generate` | 简单生成接口（纯文本 SSE 或 JSON） |
| `POST /v1/completions` | OpenAI 文本补全 |
| `POST /v1/chat/completions` | OpenAI 对话补全 |

所有生成接口均支持流式（`stream=true`）和非流式输出。暂不支持 `n > 1`。

## 架构概览

```
HTTP 请求
  → API Server（FastAPI + SSE）
  → FrontendManager → TokenizerWorker（独立进程）
  → [ZMQ 或 mp.Queue]
  → Scheduler（独立进程）
      DecodeManager  — dict[uid, Req]，decode 优先耗尽
      PrefillManager — 前缀缓存匹配、分块 prefill
      CacheManager   — 分页分配、延迟释放、H2D scatter
      RadixPrefixCache — Radix 树 + 堆式 LRU 淘汰
  → Engine
      Model（Llama / Qwen2 / Qwen3 / Qwen3-MoE / Mistral）
      KV Cache（分页 MHA）
      CUDA Graph Runner
      采样器
  → DetokenizeRequest → TokenizerWorker → StreamChunk → SSE
```

调度器默认运行重叠调度循环：第 N 批的 GPU 执行与第 N-1 批的 CPU 后处理并行进行。

### 前缀缓存核心设计

`RadixPrefixCache` 是一棵 Radix 树，每个节点存储一段前缀对应的 token ID 和 KV 页索引。

- **引用计数**：`lock_handle` / `unlock` 从叶节点向上遍历，`ref_count == 0 && is_leaf()` 的节点才可淘汰
- **堆式 LRU**：可淘汰叶节点维护在 `(timestamp, node)` 最小堆中；堆条目按需推入、弹出时懒惰校验（跳过过期条目）
- **复杂度**：淘汰 k 页的开销为 O(k log h)，而不是每次全树 DFS O(n)

### 模型工厂

Llama 系列架构（Llama、Mistral、Qwen2、Qwen3）共用同一套解码器结构，仅 Attention 参数不同。`models/_decoder.py` 提供 `build_decoder_model(attn_kwargs)` 工厂函数，每个模型文件仅需 3 行：

```python
from ._decoder import build_decoder_model
Qwen3ForCausalLM = build_decoder_model({"has_qk_norm": True})
```

## 文档导航

- [English README](README.md)
- [架构设计](docs/architecture.md) — 子系统设计、算法细节、数据流
- [开发者指南](docs/developer-guide.md) — 模块内部实现、阅读路径
- [API 参考](docs/api-guide.md) — 接口请求/响应格式
- [开发工作流](docs/development.md) — 本地运行、测试命令

## 当前状态

控制平面（API、transport、调度、缓存）已基本完整，可端到端运行。待完成工作：

- 真实 FlashAttention/FlashInfer Attention 后端（当前为 HF fallback）
- ZMQ 重叠调度在持续 TP 负载下的边界情况处理
- 更完整的 OpenAI API 兼容性（`n > 1`、更多 schema 字段）
