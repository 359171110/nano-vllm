# 02 · PagedAttention 与 KV Cache 块管理

## 学习目标

- KV Cache 为什么要切块？
- "页表 / 物理块 / slot_mapping" 分别是什么？
- 一次 attention 如何被翻译成对块池的随机访问？

## 涉及源码

- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)（`allocate_kv_cache`、`prepare_prefill`、`prepare_decode`）
- [nanovllm/layers/attention.py](../nanovllm/layers/attention.py)
- [nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)（本章只看 `Block` 数据结构；前缀缓存放下一章）

## 2.1 设计动机

朴素 KV Cache 有三个痛点：
1. 每条序列按 `max_seqlen` 预分配 → 短序列浪费严重
2. 序列结束后整块释放 → 显存碎片
3. batch 内序列不等长 → padding 浪费

vLLM 借鉴**操作系统分页**的思路：
- 物理上预分配一大块连续显存，切成固定大小的"块"（默认 256 token）
- 每条序列拥有一张"页表"（`block_table`），指向若干物理块
- 块可以被动态分配 / 回收 / 共享

## 2.2 块池的形状

[ModelRunner.allocate_kv_cache()](../nanovllm/engine/model_runner.py)：

```python
self.kv_cache = torch.empty(
    2,                      # K 与 V 各一份
    num_hidden_layers,      # 每层一份
    num_kvcache_blocks,     # 总块数（动态算出）
    block_size,             # 256 token
    num_kv_heads,
    head_dim,
)
```

`num_kvcache_blocks` 不是死写，而是基于剩余显存动态算出：

```python
free, total = torch.cuda.mem_get_info()
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
config.num_kvcache_blocks = int(total * 0.9 - used - peak + current) // block_bytes
```

> 这就是为什么 vLLM 启动时会跑一次 `warmup_model`：先用最大 batch 跑一次，把 PyTorch 内部 workspace、kernel cache 这些"额外开销"激活；之后再用剩余显存把 KV Cache 撑到极限。

随后把每个 `Attention` 模块的 `k_cache / v_cache` 引用挂到这个大张量的相应切片上：

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

## 2.3 写入路径：slot_mapping

每个新 token 需要把 K、V 写到块池的某个槽位。槽位编号是 **全局唯一** 的：

```
slot = block_table[i] * block_size + offset_in_block
```

[prepare_prefill()](../nanovllm/engine/model_runner.py) 计算每个 token 的 slot：

```python
slot_start = seq.block_table[i] * self.block_size + (start % self.block_size)
slot_mapping.extend(range(slot_start, slot_end))
```

这些 slot 编号被打包成一个 `int32` 张量，作为 attention 层的"写入地址簿"。

写入由 [store_kvcache_kernel](../nanovllm/layers/attention.py) 这个 Triton 内核完成：

```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)         # 第几个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return          # padding token（CUDA Graph 时会用），跳过
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

每个 program 处理一个 token，把它的 K、V 直接写到 slot 对应的全局位置。**没有任何 reshape、没有任何拷贝中转**。

> 真实 vLLM 用 CUDA C++ 写了等价的 `cache_kernels.cu::reshape_and_cache`，逻辑一模一样。

#### Q: slot_mapping 到底是怎么算出来的？为什么要这么算？

分三步看：是什么 → 为什么 → 怎么算。

**第一步：slot 是什么**

块池 `kv_cache` 物理形状是 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`。把后面两个 head 维度合并看，在单层、K/V 二选一的视角下，它其实是一个 `num_blocks * block_size` 个「格子」的一维数组，每个格子存 1 个 token 的 K（或 V）向量。

**slot 就是这个一维数组的下标**。

```
block 0          block 1          block 2          ...
[s0 s1 s2 s3]   [s4 s5 s6 s7]   [s8 s9 s10 s11]   ...
   block_size=4
```

**第二步：为什么需要 slot_mapping**

attention 算出来的 K/V 是「本步要算的 token 拍平」的：形状 `[N, num_kv_heads, head_dim]`（`N = Σ seqlen_q`）。但这 N 个 token 在 KV 块池里并不连续——一个序列可能占 7 号、 9 号两块，另一个可能占 4 号一块。

所以需要一张「翻译表」告诉内核：

> 第 0 个 token 该写到 slot 28，第 1 个该写到 29，第 4 个该写到 36 ……

这张翻译表就是 `slot_mapping`。形状与 token 数一致：`[N]`。

**第三步：怎么算**

公式只有一行：

```
slot = block_table[i] * block_size + offset_in_block
```

拆开看：
- `block_table[i]`：该 token 所在「虚拟块」对应的**物理块号**（序列的页表查出来的）
- `* block_size`：跳过前面 N 个块占据的 slot，到达本块的起点
- `+ offset_in_block`：加上该 token 在本块内的偏移

**一个完整例子**

假设 `block_size = 4`，序列 A 总长 6，`block_table = [7, 9]`（占了物理块 7 与 9）。今天是一次完整 prefill（`num_cached_tokens=0`，`num_scheduled_tokens=6`）：

```
token 索引  | 虚拟块 i | 块内偏移 | 物理块号 | slot 计算         | slot
----------+----------+----------+----------+--------------------+------
0         | 0        | 0        | 7        | 7*4 + 0           | 28
1         | 0        | 1        | 7        | 7*4 + 1           | 29
2         | 0        | 2        | 7        | 7*4 + 2           | 30
3         | 0        | 3        | 7        | 7*4 + 3           | 31
4         | 1        | 0        | 9        | 9*4 + 0           | 36
5         | 1        | 1        | 9        | 9*4 + 1           | 37
```

所以 `slot_mapping = [28, 29, 30, 31, 36, 37]`。中间跳过了 32~35，因为那 4 个格子属于物理块 8，不是本序列的。

**代码里为什么看起来这么绕**

代码按**虚拟块**一块一块地处理，而不是一个 token 一个 token 地调 append，是为了让 `slot_mapping.extend(range(slot_start, slot_end))` 一句能查出连续多个 slot（块内是连续的）。三个边界变量：

```python
start_block = start // block_size                 # 本步起点落在哪个虚拟块
end_block   = (end + block_size - 1) // block_size# 本步终点落在哪个虚拟块后一个
for i in range(start_block, end_block):
    slot_start = block_table[i] * block_size      # 块 i 的 slot 起点
    if i == start_block:
        slot_start += start % block_size           # 起始块：从块内偏移开始
    if i != end_block - 1:
        slot_end = block_table[i] * block_size + block_size   # 中间块：一整块
    else:
        slot_end = block_table[i] * block_size + end - i * block_size  # 终止块：只到 end
    slot_mapping.extend(range(slot_start, slot_end))
```

这三个分支就是为 chunked prefill / 前缀缓存准备的。举个 chunked 例子：序列 A 已缓存 `start=2` 个 token，本步再算 `seqlen_q=4`（`end=6`）：

- start_block = `2 // 4 = 0`，end_block = `6+3)//4 = 2`
- i=0（起始块）：slot_start = `7*4 + 2 = 30`，slot_end = `7*4 + 4 = 32` → `[30, 31]`
- i=1（终止块）：slot_start = `9*4 = 36`，slot_end = `9*4 + 6 - 1*4 = 38` → `[36, 37]`
- 结果 `slot_mapping = [30, 31, 36, 37]`，正好对应本步要写入的 4 个 token

**内核里怎么用 slot**

[store_kvcache_kernel](../nanovllm/layers/attention.py) 里：

```python
idx  = tl.program_id(0)               # 第几个 token，对应 slot_mapping 的下标
slot = tl.load(slot_mapping_ptr + idx)# 查表拿到该 token 该写到哪个 slot
cache_offsets = slot * D + tl.arange(0, D)   # D = num_kv_heads * head_dim
tl.store(k_cache_ptr + cache_offsets, key)
```

把 `k_cache` 看作一维数组（连续存储），`slot * D` 就是这个 slot 的起始字节（元素）偏移。于是一条「从 token 下标到全局内存位置」的路径就闭环了。

**一句话总结**

> `slot_mapping[i] = block_table[i 所在虚拟块] * block_size + i 在块内的偏移`。
> 它是一张「token 下标 → KV 块池全局下标」的翻译表，让内核不需要任何 reshape 、不需要任何拷贝，一句 `tl.store` 就能把 K/V 落到正确位置。

## 2.4 读取路径：block_table 直接喂给 FlashAttention

读取没必要自己写内核，因为 FlashAttention ≥ 2.5 已经原生支持 paged KV：

```python
# layers/attention.py
if context.is_prefill:
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        block_table=context.block_tables,   # ← 关键
        causal=True,
    )
else:
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,   # ← 关键
    )
```

`block_table` 是一个 `[batch_size, max_num_blocks]` 的 `int32` 张量，FlashAttention 内部会按它去 `k_cache / v_cache` 里 gather 真正的 KV。

## 2.5 一个端到端例子

假设 `block_size = 4`，batch 内有两条序列：

```
seq A: prompt = [t0,t1,t2,t3,t4,t5]   ← 6 个 token
seq B: prompt = [t0,t1,t2]            ← 3 个 token
```

经 Scheduler 后可能拿到：
- A 占 2 块（block_id=7、9）
- B 占 1 块（block_id=4）

`block_tables` 张量：
```
[[7, 9],
 [4, -1]]   # B 不够长，后面 padding -1
```

写入 KV 时：
- A 的 token0~3 → slot `7*4+(0..3) = 28..31`
- A 的 token4~5 → slot `9*4+(0..1) = 36..37`
- B 的 token0~2 → slot `4*4+(0..2) = 16..18`

读取时，FlashAttention 看到 `block_tables` 就知道 A 的 KV 散布在块 7、9，自动 gather。

## 2.6 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| K/V 用一张 6D 大张量 | 同结构 |
| 写入用 Triton 内核 | CUDA C++ kernel |
| 读取直接调 flash-attn | 历史上有自家 `paged_attention_v1/v2`，新版多用 FlashInfer |
| 块大小固定 256 | 可配置，常见 16/32 |
| 不支持 swap-to-CPU | 支持 |

## 2.7 动手任务

1. 在 [allocate_kv_cache](../nanovllm/engine/model_runner.py) 末尾打印 `num_kvcache_blocks`，把 `gpu_memory_utilization` 从 0.9 改到 0.5，对比块数变化
2. 把 `kvcache_block_size` 改成 512（注意 `__post_init__` 中要求 `% 256 == 0`），观察 prefill 吞吐变化；再改成 256，观察前缀缓存命中率
3. 思考：为什么 `slot_mapping` 用 `int32` 而不是 `int64`？（提示：考虑总 token 数与显存带宽）

---

上一章 ← [01 · 整体架构](./01-architecture.md) | 下一章 → [03 · 前缀缓存](./03-prefix-caching.md)
