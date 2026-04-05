# bench — End-to-End Benchmark Suite

对比 radixInfer、vLLM、SGLang 的推理性能，并生成可直接嵌入 README 的图表。

## 文件说明

| 文件 | 说明 |
|---|---|
| `bench_e2e.py` | 核心压测脚本（无第三方依赖，纯标准库） |
| `plot_results.py` | 读取 JSON 结果，生成 PNG 对比图（需要 matplotlib） |
| `run_bench.sh` | 一键运行所有测试场景并生成图表 |

## 测试场景

### Scenario 1 — 并发扫描（三引擎对比）

固定输入/输出 token 数，扫描并发度 `[1, 4, 8, 16, 32]`，对比三个引擎的：
- 输出吞吐（tokens/s）
- TTFT（首 token 延迟）
- TPOT（每 token 耗时）
- 端到端延迟 p50/p90/p99

### Scenario 2 — 输入长度扫描（radixInfer）

固定并发=16，输出=256 tokens，扫描输入长度 `[128, 512, 1024, 2048]`，观察长上下文下的吞吐变化。

### Scenario 3 — 前缀缓存命中效果（radixInfer 专项）

所有请求共享一个 400-token 公共前缀，对比有/无缓存命中时的吞吐和 TTFT 差异。

## 快速开始

### 1. 安装依赖

```bash
pip install matplotlib
```

### 2. 启动各引擎

```bash
# radixInfer (port 1919)
conda activate sglang
PYTHONPATH=python python -m radixinfer --model Qwen/Qwen3-0.6B --device cuda:0 --port 1919

# vLLM (port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B --port 8000 --dtype bfloat16

# SGLang (port 30000)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B --port 30000 --dtype bfloat16
```

### 3. 一键运行

```bash
cd /path/to/radixInfer

MODEL=Qwen/Qwen3-0.6B \
NUM_REQUESTS=200 \
bash bench/run_bench.sh
```

结果写入 `bench/results/`，图表写入 `bench/results/plots/`。

### 4. 单独绘图

```bash
# 三引擎并发=16 快照对比
python3 bench/plot_results.py \
    bench/results/vllm_c16.json \
    bench/results/sglang_c16.json \
    bench/results/radixinfer_c16.json \
    --out-dir bench/results/plots

# 并发扫描折线图
python3 bench/plot_results.py \
    --sweep bench/results/radixinfer_c*.json \
    --sweep-key concurrency \
    --out-dir bench/results/plots/radixinfer_sweep

# 前缀缓存效果
python3 bench/plot_results.py \
    --prefix-cache \
    bench/results/radixinfer_c16_nocache.json \
    bench/results/radixinfer_c16_cache.json \
    --out-dir bench/results/plots/prefix_cache
```

## 手动压测单个引擎

```bash
# 已启动的服务
python3 bench/bench_e2e.py \
    --engine generic \
    --base-url http://127.0.0.1:1919/v1 \
    --model Qwen/Qwen3-0.6B \
    --concurrency 16 \
    --num-requests 200 \
    --input-tokens 512 \
    --output-tokens 256 \
    --result-json bench/results/radixinfer_c16.json

# 自动启动服务再压测
python3 bench/bench_e2e.py \
    --engine vllm \
    --server-command "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --port 8000" \
    --base-url http://127.0.0.1:8000/v1 \
    --ready-timeout 300 \
    --concurrency 16 \
    --num-requests 200 \
    --result-json bench/results/vllm_c16.json
```

## 关键参数

| 参数 | 说明 |
|---|---|
| `--engine` | `vllm` / `sglang` / `generic`（radixInfer 用 generic） |
| `--base-url` | 引擎的 OpenAI 兼容 API 地址 |
| `--concurrency` | 并发请求数 |
| `--num-requests` | 总请求数 |
| `--input-tokens` | 输入长度，支持范围如 `512:1024` |
| `--output-tokens` | 输出长度，支持范围如 `128:256` |
| `--warmup-requests` | 预热请求数（不计入结果） |
| `--result-json` | 结果保存路径（JSON） |
| `--result-jsonl` | 逐请求明细保存路径 |
| `--dataset` | 使用 JSONL 数据集替代合成 prompt |

## 测试环境

| 项目 | 值 |
|---|---|
| GPU | 2× NVIDIA RTX 4090 D (24GB) |
| 模型 | Qwen/Qwen3-0.6B（对比用）|
| 输入长度 | 512 tokens |
| 输出长度 | 256 tokens |
| 请求数 | 200 |
| 预热 | 8 requests |
