# Nano-vLLM 源码学习教程

> 用约 1200 行 Python 看懂真实 vLLM 的核心机制。

## 这份教程是什么

不是逐行注释，而是按 **"自顶向下 → 关键内核 → 优化技巧"** 的顺序，把 nano-vllm 当作真实 vLLM 的"骨架地图"读一遍。每章控制在 30~45 分钟阅读量。

读完之后你应该能：

- 看懂 vLLM 主循环的真正职责切分
- 解释 PagedAttention 与前缀缓存的工作原理
- 复述 Continuous Batching、Chunked Prefill、Preemption 的边界条件
- 把张量并行 / CUDA Graph / KV 块化几个独立优化串成一张图
- 知道真实 vLLM 在哪些维度做了 nano-vllm 没做的扩展

## 章节索引

| 章节 | 主题 | 关键文件 |
|---|---|---|
| [01](./01-architecture.md) | 整体架构与请求生命周期 | `engine/llm_engine.py`、`engine/sequence.py` |
| [02](./02-paged-attention.md) | PagedAttention 与 KV Cache 块管理 | `engine/model_runner.py`、`layers/attention.py` |
| [03](./03-prefix-caching.md) | 前缀缓存：把 KV 变成内容寻址缓存 | `engine/block_manager.py` |
| [04](./04-scheduler.md) | 调度器：Continuous Batching + Chunked Prefill + Preempt | `engine/scheduler.py` |
| [05](./05-prefill-vs-decode.md) | Prefill 与 Decode 的两套数据通路 | `engine/model_runner.py`、`utils/context.py` |
| [06](./06-tensor-parallel.md) | 张量并行（Megatron-style） | `layers/linear.py`、`layers/embed_head.py` |
| [07](./07-cuda-graph.md) | CUDA Graph 加速 Decode | `engine/model_runner.py` |
| [08](./08-sampler-and-loader.md) | 采样器与权重加载 | `layers/sampler.py`、`utils/loader.py` |
| [09](./09-vs-real-vllm.md) | 与真实 vLLM 的差异和进阶路线 | — |

## 每章固定结构

1. **学习目标**：本章读完应该能回答的问题
2. **涉及源码**：本章会精读的文件
3. **设计动机**：vLLM 为什么这么做（从 GPU 特性反推）
4. **代码精读**：逐段拆解关键代码
5. **与真实 vLLM 的对比**：nano-vllm 简化了什么
6. **动手任务**：本机就能跑的小练习

## 前置要求

- Python 3.10–3.12，熟悉 PyTorch
- 大致了解 Transformer / Attention / LayerNorm 的数学形式
- 跑过一次 `example.py`（模型权重已下载）

## 开始之前

强烈建议先把项目跑起来再读章节：

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/

python example.py
```

跑通之后，从 [第 01 章](./01-architecture.md) 开始。
