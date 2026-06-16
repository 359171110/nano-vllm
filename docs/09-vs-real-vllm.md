# 09 · 与真实 vLLM 的差异和进阶路线

## 学习目标

- nano-vllm 简化掉了真实 vLLM 的哪些重要能力？
- 想接驳真实 vLLM 源码，应该按什么顺序读？
- vLLM 当下还在演进哪些方向？

## 9.1 一张全景对比表

| 维度 | nano-vllm | 真实 vLLM |
|---|---|---|
| 入口 | `LLM.generate(prompts, params)` 同步 | `LLM` + `AsyncLLMEngine`，含 OpenAI 兼容 HTTP server |
| 流式输出 | ❌ 一次性返回 | ✅ token-level streaming |
| 采样能力 | 仅温度 | top-k / top-p / min-p / 频率/存在惩罚 / logit bias / logprobs |
| Beam Search / n>1 | ❌ | ✅ |
| Speculative Decoding | ❌ | ✅（draft model + accept/reject） |
| Guided Decoding | ❌ | ✅（XGrammar / outlines） |
| LoRA / Multi-LoRA | ❌ | ✅（punica kernel） |
| 量化 | ❌ | ✅（GPTQ / AWQ / FP8 / Marlin） |
| 抢占模式 | 仅 recompute | recompute + swap-to-CPU |
| 并行 | TP | TP + PP + EP（专家并行） |
| 多模态 | ❌ | ✅（vision encoder 输出注入 KV） |
| 调度策略 | FCFS | FCFS / priority |
| 块大小 | 固定 256 | 可配（常见 16/32） |
| Attention backend | flash-attn 单一 | FlashAttn / xFormers / FlashInfer / TorchSDPA / Pallas（TPU） |
| 控制流通信 | SharedMemory + Event | Ray RPC（默认）或 ZMQ |
| Disaggregated serving | ❌ | ✅（prefill / decode 分离部署） |

## 9.2 学习路径建议

### 第一阶段：吃透 nano-vllm

- 跑通 `example.py` 与 `bench.py`
- 每章动手任务都做一遍
- 能向别人讲清楚"一个 token 的诞生过程"

### 第二阶段：对照 vLLM 源码

按这个顺序读最顺：

```
vllm/engine/llm_engine.py          ← 对照 nanovllm/engine/llm_engine.py
vllm/core/scheduler.py             ← 对照 nanovllm/engine/scheduler.py
vllm/core/block_manager_v2.py      ← 对照 nanovllm/engine/block_manager.py
vllm/worker/model_runner.py        ← 对照 nanovllm/engine/model_runner.py
vllm/attention/                    ← 对照 nanovllm/layers/attention.py
vllm/model_executor/layers/        ← 对照 nanovllm/layers/
```

你会发现函数名、参数名、状态机都对得上——nano-vllm 是 vLLM 的"真子集"。

### 第三阶段：研究 V1 引擎重写

2024 下半年 vLLM 团队推出 V1 引擎，主要改进：
- **零拷贝调度**：`SchedulerOutputs` 用张量代替 Python 对象
- **异步化**：调度与执行解耦
- **Chunked prefill 与 decode 真正混合**
- **更好的 LoRA / 多模态集成**

代码在 `vllm/v1/`，与 V0 共存一段时间。

### 第四阶段：阅读论文 / RFC

- 原始论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)
- Continuous Batching：[Orca (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu)
- Speculative Decoding：[Fast Inference from Transformers via Speculative Decoding (ICML'23)](https://arxiv.org/abs/2211.17192)
- Disaggregated serving：[DistServe (OSDI'24)](https://arxiv.org/abs/2401.09670)
- vLLM 自身的 RFC：见 [GitHub Discussions](https://github.com/vllm-project/vllm/discussions)

## 9.3 推荐的"造轮子"路线

读完之后想动手实现自己的版本？建议按这个顺序加 feature：

| 阶段 | 加什么 | 难度 | 收获 |
|---|---|---|---|
| 1 | 流式输出（生成器接口） | ★ | 理解请求生命周期 |
| 2 | top-k / top-p 采样 | ★ | 熟悉算子级优化 |
| 3 | logprobs 返回 | ★★ | 理解模型最后一层布局 |
| 4 | swap 抢占模式 | ★★ | 深入 KV Cache 内存模型 |
| 5 | LoRA 加载 + adapter forward | ★★★ | 理解 punica kernel 的位置 |
| 6 | Speculative Decoding | ★★★★ | 理解 vLLM 调度的灵活性极限 |
| 7 | 量化推理（FP8 / AWQ） | ★★★★ | 进入 CUDA / Marlin kernel 世界 |

## 9.4 vLLM 当下的演进方向

### Disaggregated Serving

prefill 是 compute-bound、decode 是 memory-bound——把它们部署在不同硬件上（甚至不同集群）会比混合部署更划算。需要解决跨节点 KV Cache 传输。

### Prefix Cache 持久化

当前前缀缓存只在引擎运行时存在。让它持久化到磁盘 / 共享 KV 服务器，多实例可以复用，对系统 prompt 的多用户场景收益巨大。

### 长上下文（128K+）

传统 attention 在长上下文下会 OOM。需要 chunked prefill、稀疏 attention、KV 量化等多种技术配合。

### 异构硬件

TPU、AMD、Inferentia、华为昇腾……每种硬件的 attention backend 都需要适配。

### Multi-step Decoding / Continuous Speculative Decoding

一次前向算多个 token，进一步压榨 decode 阶段的 GPU 利用率。

## 9.5 一个简短的总结

把整个教程压成一句话：

> **vLLM 的本质是把"操作系统的分页管理"搬到了 GPU 显存上，用块化的 KV Cache + Continuous Batching 调度，配合 FlashAttention / 张量并行 / CUDA Graph 这些独立优化，端到端把 LLM 推理的吞吐推到极限。**

恭喜你读完整个教程。现在你应该能：

- 在面试中讲清楚 vLLM 的核心机制
- 给真实 vLLM 提 issue 或 PR
- 设计自己的推理引擎或者推理优化方案

---

上一章 ← [08 · 采样器与权重加载](./08-sampler-and-loader.md) | 回到 [目录](./README.md)
