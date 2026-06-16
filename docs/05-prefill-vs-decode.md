# 05 · Prefill 与 Decode 的两套数据通路

## 学习目标

- prefill 与 decode 在张量布局上有什么差异？
- `Context` 这个全局对象为什么是必需的？
- LM Head 在 prefill 阶段如何"只算最后一个位置"省下大量算力？

## 涉及源码

- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)（`prepare_prefill`、`prepare_decode`、`run_model`）
- [nanovllm/utils/context.py](../nanovllm/utils/context.py)
- [nanovllm/layers/attention.py](../nanovllm/layers/attention.py)
- [nanovllm/layers/embed_head.py](../nanovllm/layers/embed_head.py)

## 5.1 全局 Context：算子层的"暗号"

[context.py](../nanovllm/utils/context.py) 极其简单：

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()
def get_context(): return _CONTEXT
def set_context(...): ...
def reset_context(): ...
```

为什么不通过函数参数传？因为模型是标准 `nn.Module`：

```python
def forward(self, input_ids, positions): ...
```

如果要让每个 attention 层都拿到 `slot_mapping / block_tables / cu_seqlens` 这堆参数，要么改签名，要么走全局对象。**全局对象写起来最简洁，副作用也最容易约束**——前向之前 `set_context`，前向之后 `reset_context`。

> 真实 vLLM 是同款思路，叫 `AttentionMetadata`，通过线程局部变量传递。

## 5.2 Prefill 数据通路

[prepare_prefill()](../nanovllm/engine/model_runner.py) 要为变长 batch 准备：

| 张量 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `[Σ seqlen_q]` | 所有序列的 token 拍平 |
| `positions` | `[Σ seqlen_q]` | 每个 token 在自己序列里的位置 |
| `cu_seqlens_q` | `[batch+1]` | query 累计长度（FlashAttention 变长接口） |
| `cu_seqlens_k` | `[batch+1]` | key 累计长度（含已缓存前缀） |
| `slot_mapping` | `[Σ seqlen_q]` | 每个新 token 的写入 slot |
| `block_tables` | `[batch, max_blocks]` | 仅当存在前缀缓存时填充 |

关键代码：

```python
for seq in seqs:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end                          # ← key 长度 = 起点 + 本步要算的
    input_ids.extend(seq[start:end])
    positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    ...
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:     # 有 prefix cache 命中
    block_tables = self.prepare_block_tables(seqs)
```

`seqlen_k > seqlen_q` 当且仅当有前缀已经在 KV Cache 里——此时 attention 走 paged 路径（用 `block_tables`）；否则走 dense 路径（K/V 直接来自当前 batch）。

## 5.3 Decode 数据通路

[prepare_decode()](../nanovllm/engine/model_runner.py) 简单得多：

```python
for seq in seqs:
    input_ids.append(seq.last_token)
    positions.append(len(seq) - 1)
    context_lens.append(len(seq))
    slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

| 张量 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `[batch]` | 每条序列上一步的 token |
| `positions` | `[batch]` | 每条序列当前位置 |
| `slot_mapping` | `[batch]` | 新 token 写入位置（最后一块的最后一格） |
| `context_lens` | `[batch]` | 每条序列总长度 |
| `block_tables` | `[batch, max_blocks]` | 必填，FlashAttention 用来 gather KV |

decode 一定走 paged 路径，所以 `block_tables` 永远存在。

## 5.4 Attention 算子的两个分支

[Attention.forward()](../nanovllm/layers/attention.py)：

```python
def forward(self, q, k, v):
    context = get_context()
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)   # 公共：写入 KV

    if context.is_prefill:
        if context.block_tables is not None:        # prefix cache 命中
            k, v = k_cache, v_cache                 # 走 paged 模式
        o = flash_attn_varlen_func(q, k, v,
            cu_seqlens_q=..., cu_seqlens_k=..., block_table=...)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables)
    return o
```

三个关键判断：
1. `k_cache.numel()`：warmup 时还没分配 KV Cache，跳过写入
2. `context.is_prefill`：prefill 用变长接口，decode 用单 token 接口
3. `context.block_tables is not None`：prefill 阶段是否需要走 paged 模式

## 5.5 LM Head：prefill 的省算技巧

[ParallelLMHead.forward()](../nanovllm/layers/embed_head.py)：

```python
def forward(self, x):
    context = get_context()
    if context.is_prefill:
        last_indices = context.cu_seqlens_q[1:] - 1   # 每条序列最后一个 token 的位置
        x = x[last_indices].contiguous()
    logits = F.linear(x, self.weight)
    ...
```

prefill 阶段，hidden states 形如 `[Σ seqlen_q, hidden]`。但 **采样只关心每条序列最后一个 token 的 logits**——前面的位置算 logits 是浪费。

`cu_seqlens_q = [0, l1, l1+l2, ...]`，所以 `cu_seqlens_q[1:] - 1` 正好是每条序列最后位置在拍平张量里的下标。

> 假设 batch 4 条序列，长度 1024 / 800 / 600 / 400，`hidden=4096`、`vocab=152064`。
> 不省算：`(1024+800+600+400) × 4096 × 152064 ≈ 1.8 TFLOPs`
> 省算后：`4 × 4096 × 152064 ≈ 2.5 GFLOPs`
> **三个数量级的差距**。

## 5.6 一次完整调用流程

```
ModelRunner.run(seqs, is_prefill)
   │
   ├─ prepare_prefill / prepare_decode
   │     ├─ 计算 input_ids / positions / slot_mapping
   │     ├─ 计算 cu_seqlens_q/k 或 context_lens
   │     ├─ 计算 block_tables（如需要）
   │     └─ set_context(...)
   │
   ├─ run_model(input_ids, positions, is_prefill)
   │     └─ Qwen3 forward:
   │         embed_tokens →
   │         N × (RMSNorm → QKV → RoPE → Attention → O → MLP) →
   │         Norm → LM Head
   │           Attention.forward 内部:
   │             store_kvcache_kernel(slot_mapping)
   │             flash_attn_varlen_func / flash_attn_with_kvcache
   │
   ├─ Sampler(logits, temperatures) → token_ids
   │
   └─ reset_context()
```

## 5.7 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| 全局 `_CONTEXT` 单例 | `AttentionMetadata` 通过 thread-local 传递 |
| 一种 attention backend | FlashAttn / xFormers / FlashInfer / TorchSDPA |
| LM Head prefill 取最后位置 | 同样优化 |
| 模型 forward 签名最小 | 多了 `intermediate_tensors`（PP 用） |

## 5.8 动手任务

1. 在 `Attention.forward` 入口处 `print(context.is_prefill, q.shape, k.shape)`，观察两阶段形状差异
2. 把 LM Head 中"只取最后位置"的优化注释掉，跑 `bench.py`，对比 prefill 吞吐
3. 思考：为什么 decode 阶段 `q` 还要 `unsqueeze(1)`？
   *答：`flash_attn_with_kvcache` 接口要求 q 形状 `[batch, seqlen_q, num_heads, head_dim]`，decode 时 seqlen_q=1。*

---

上一章 ← [04 · 调度器](./04-scheduler.md) | 下一章 → [06 · 张量并行](./06-tensor-parallel.md)
