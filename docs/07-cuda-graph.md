# 07 · CUDA Graph 加速 Decode

## 学习目标

- CUDA Graph 解决了什么问题？
- 为什么要"桶化"batch size？
- `graph_vars` 这些固定 buffer 是怎么用的？

## 涉及源码

- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)（`capture_cudagraph`、`run_model`）

## 7.1 设计动机

decode 阶段每步只算 1 个 token，但整张前向却要跑 N 层 transformer。每一层调用一堆 PyTorch 算子，每个算子都会经历：
1. Python 调度（解释器开销）
2. 检查参数、构造 CUDA kernel launch
3. 把 launch 命令推到 CUDA 流

这些 host 端开销在小算量场景下 **占比惊人**——经常超过 GPU 实际计算时间。

CUDA Graph 把整个前向"录制"成一张图，重放时跳过所有 host 端开销，只让 GPU 按预录的 kernel 序列执行。

## 7.2 核心限制

CUDA Graph 要求 **每次调用的 tensor 地址、形状完全一致**。这带来两个挑战：

1. batch size 每步可能变化 → 用桶化 + padding 解决
2. 输入数据每步不同 → 用固定 buffer + 拷贝解决

## 7.3 桶化 batch size

[capture_cudagraph()](../nanovllm/engine/model_runner.py)：

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

不可能为每个 batch size 都录一张图（显存爆炸），所以选有限离散值：1、2、4、8、16、32、48、…、512。

运行时向上取整 padding：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

例如 `bs=37`，就用 `bs=48` 的图，多出来的 11 个位置 padding。

## 7.4 录制阶段

```python
input_ids    = torch.zeros(max_bs, dtype=torch.int64)
positions    = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs      = torch.zeros(max_bs, hidden_size)

for bs in reversed(self.graph_bs):                    # 从大到小，复用 pool
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # capture
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = graph

self.graph_vars = dict(
    input_ids=input_ids, positions=positions,
    slot_mapping=slot_mapping, context_lens=context_lens,
    block_tables=block_tables, outputs=outputs,
)
```

三个关键技巧：

### 7.4.1 共享 graph pool

```python
with torch.cuda.graph(graph, self.graph_pool):
```

所有图共享同一块中间显存池。每张图录制的算子在 pool 里有自己的内存布局，重放时不互相干扰；不共享的话每张图都自留一份中间 buffer，显存爆炸。

### 7.4.2 倒序录制

```python
for bs in reversed(self.graph_bs):
```

先录最大 batch（48 → … → 1）。第一次录制时 `self.graph_pool=None`，由它建立 pool；之后小图复用同一个 pool。先大后小可以让 pool 一次性分配到位，避免后续动态扩容。

### 7.4.3 warmup 在 capture 之前

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
with torch.cuda.graph(graph, self.graph_pool):
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
```

warmup 让 PyTorch 把 lazy 算子初始化、kernel 选型、cuBLAS workspace 分配等都跑完一遍；capture 才是干净的 kernel launch 序列。

## 7.5 重放阶段

[run_model()](../nanovllm/engine/model_runner.py)：

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs]    = input_ids        # 拷贝实际数据到固定 buffer
        graph_vars["positions"][:bs]    = positions
        graph_vars["slot_mapping"].fill_(-1)               # 重置整张表
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()                  # 重置整张表
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

三个细节：

1. **prefill 不走 graph**：prefill 形状高度多变，桶化代价太大
2. **`slot_mapping` 重置为 -1**：让 padding 部分的 KV 写入在 [store_kvcache_kernel](../nanovllm/layers/attention.py) 中被 `if slot == -1: return` 跳过，避免污染 KV Cache
3. **`context_lens` 清零**：让 padding 部分的 attention 长度为 0，FlashAttention 自动跳过

## 7.6 性能直觉

在小 batch decode 场景：

| 模式 | host 开销占比 | 典型加速 |
|---|---|---|
| eager | 50%~70% | 1× |
| `torch.compile` | 30%~50% | 1.2~1.5× |
| CUDA Graph | < 5% | 1.5~2.5× |

vLLM 默认开启 CUDA Graph，关闭它（`enforce_eager=True`）通常用于：
- debug
- 模型本身有动态控制流
- batch size 极不规则

## 7.7 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| 桶化 [1,2,4,8,16,32,…,512] | 同思路，桶位可配 |
| 单一 graph_pool | 同 |
| prefill 不走 graph | 同（V0）；V1 引擎尝试把 chunked prefill 也放进 graph |
| 仅 decode | LoRA 时还要为 LoRA-on/off 各录一组 |

## 7.8 动手任务

1. `enforce_eager=True` 与 `False` 各跑一次 `bench.py`，对比 decode 吞吐
2. 把 `graph_bs` 改成 `[1, 8, 64, 512]`（粗粒度桶），观察吞吐下降——padding 浪费
3. 思考：为什么 `block_tables` 不需要 `fill_(-1)`？
   *答：因为 attention 是按 `context_lens` 来读 KV 的，`context_lens=0` 的 padding 行根本不会去访问 `block_tables`。*

---

上一章 ← [06 · 张量并行](./06-tensor-parallel.md) | 下一章 → [08 · 采样器与权重加载](./08-sampler-and-loader.md)
