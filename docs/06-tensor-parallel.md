# 06 · 张量并行（Megatron-style）

## 学习目标

- 列并行 / 行并行的数学含义是什么？
- Attention block 与 MLP block 各需要几次 all-reduce？
- 多进程是怎么协同的？为什么用 `SharedMemory + Event`？

## 涉及源码

- [nanovllm/layers/linear.py](../nanovllm/layers/linear.py)
- [nanovllm/layers/embed_head.py](../nanovllm/layers/embed_head.py)
- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)（`call`、`read_shm`、`write_shm`、`loop`）
- [nanovllm/engine/llm_engine.py](../nanovllm/engine/llm_engine.py)（`__init__` 多进程启动）

## 6.1 Megatron 切法的核心思想

矩阵乘 `Y = X·W`，把 `W` 沿不同维度切：

| 切法 | 切的维度 | 是否需要 all-reduce | 串联中产生通信 |
|---|---|---|---|
| **列并行** | output（列） | 不需要 | 后面接行并行可消通信 |
| **行并行** | input（行） | 必须 | 通信 1 次 |

把 attention 的 `qkv_proj` 用列并行、`o_proj` 用行并行；MLP 的 `gate_up_proj` 用列并行、`down_proj` 用行并行——**整个 attention/MLP block 只通信 1 次**。这就是 Megatron-LM 论文的经典套路。

#### Q: 这个“列并行接行并行只通信 1 次”具体是怎么操作的？

下面用 Attention block 举例（MLP 完全同理），假设 `tp_size = 2`（两张卡）。

**先看不切的情况（单卡）：**

首先你要知道 Transformer 的 Attention 层有三个权重矩阵：W_q、W_k、W_v，它们把输入 x 分别投影成 query、key、value。

vLLM 为了效率，把这三个矩阵**纵向拼接成一个大矩阵** `W_qkv`，一次 GEMM 算完，再把结果沿输出维度切开：

```
假设 hidden_size=8, num_heads=4, head_dim=2:

W_q: [8, 8]   ─┐
 W_k: [8, 8]    ├── 拼接成 ──▶ W_qkv: [8, 24]
W_v: [8, 8]   ─┘

qkv = x @ W_qkv            # x:[batch, 8] @ W_qkv:[8, 24] = qkv:[batch, 24]
q, k, v = split(qkv, [8, 8, 8])  # 沿最后一维切成三段
                                   # q:[batch, 8], k:[batch, 8], v:[batch, 8]
```

实际代码就是：

```python
# nanovllm/models/qwen3.py 第 77-78 行
qkv = self.qkv_proj(hidden_states)                            # 一次 GEMM
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 沿输出维切开
```

所以 `q, k, v = split(qkv)` 就是“把大 GEMM 的结果沿输出维度切成 q、k、v 三段”。

**为什么要拼在一起再切？**

因为 GPU 喜欢大矩阵乘。做 1 次 `[8, 24]` 的大 GEMM，比做 3 次 `[8, 8]` 的小 GEMM 快得多（GPU 利用率更高）。

---

现在回到张量并行的例子：

```
单卡完整流程：

输入 x: [batch, 8]
     │
     ▼  x @ W_qkv [8, 24]
qkv: [batch, 24]
     │
     ▼  split 成三段
q: [batch, 8]    k: [batch, 8]    v: [batch, 8]
     │                                │
     ▼  attention(q, k, v)             │
o: [batch, 8]                         │
     │                                │
     ▼  o @ W_o [8, 8]                │
output: [batch, 8]
```

**现在切成 2 张卡：**

步骤 1：`qkv_proj` 用**列并行** —— 把 W_qkv 沿输出维切一刀

```
          原始 W_qkv [8, 24]
卡 0 拿到：W_qkv_0 [8, 12]   # 前半列
卡 1 拿到：W_qkv_1 [8, 12]   # 后半列

卡 0: qkv_0 = x @ W_qkv_0  → [batch, 12]  # 各算各的，无通信
卡 1: qkv_1 = x @ W_qkv_1  → [batch, 12]
```

结果：卡 0 拿到了 q/k/v 的前一半 head，卡 1 拿到后一半 head。
**无需通信！** 因为 attention 是按 head 独立计算的，每张卡算自己那几个 head 就行。

步骤 2：Attention 计算（各卡独立）

```
卡 0: o_0 = attention(q_0, k_0, v_0)  → [batch, 4]  # 各算各的
卡 1: o_1 = attention(q_1, k_1, v_1)  → [batch, 4]
```

步骤 3：`o_proj` 用**行并行** —— 把 W_o 沿输入维切一刀

```
          原始 W_o [8, 8]
卡 0 拿到：W_o_0 [4, 8]   # 前半行——刚好对应 o_0 的尺寸！
卡 1 拿到：W_o_1 [4, 8]   # 后半行

卡 0: y_0 = o_0 @ W_o_0  → [batch, 8]  # 部分结果
卡 1: y_1 = o_1 @ W_o_1  → [batch, 8]  # 部分结果
```

但这里每张卡的 `y_i` 只是最终结果的一部分（因为矩阵乘被切开了）。完整结果 = `y_0 + y_1`。

步骤 4：**唯一一次 all-reduce**

```
output = all_reduce(y_0, y_1)   # 各卡把自己的部分结果加在一起
       = y_0 + y_1              # 每张卡都拿到完整的 output
```

→ 整个 Attention block 从头到尾只要了 **1 次 all-reduce**。

**为什么列并行接行并行刚好能配合？**

关键在于“列并行的输出”和“行并行的输入”形状刚好匹配：

```
列并行输出: 卡 0 拿 [batch, output/2]，卡 1 拿 [batch, output/2]
                          ↓
行并行输入: 卡 0 要 [batch, input/2]，卡 1 要 [batch, input/2]
```

列并行的输出已经是分片的——刚好直接喂给行并行作为分片输入，**中间不需要任何通信**。只有行并行的输出才需要 all-reduce。

**MLP 完全同理：**

```
gate_up_proj (列并行, 无通信)
     ↓
SiLU & Mul (各卡独立)
     ↓
down_proj (行并行)
     ↓
all-reduce ←←← 唯一一次通信
```

**总结：一个 Transformer Layer 的通信次数**

```
Attention block:  qkv_proj(列,无) → attn(无) → o_proj(行) → all-reduce  ← 1次
MLP block:        gate_up(列,无) → act(无) → down(行) → all-reduce     ← 1次
合计: 每层 2 次 all-reduce
```

如果不用这个套路，最朴素的做法是每个线性层后都 all-reduce 一次（qkv 后一次、o_proj 后一次、gate_up 后一次、down 后一次 = 4 次）。Megatron 套路减少一半通信。

## 6.2 ColumnParallelLinear：切输出

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param, loaded_weight):
        # 从 HF 完整权重中切出本卡那一片
        shard_size = param.size(0)
        start_idx  = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)   # 不需要 all-reduce
```

每张卡的 `weight` 形状 `[output_size/tp_size, input_size]`，输出形状 `[*, output_size/tp_size]`——天然分片，无需通信。

## 6.3 RowParallelLinear：切输入

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)               # ← 必须 all-reduce
        return y
```

每张卡只看到 `input_size/tp_size` 列的 W，所以每张卡算的是 `Σ_partial`，最后 all-reduce 求和。

注意 `bias` 只在 rank 0 加，避免被加 `tp_size` 次。

## 6.4 QKV 与 Gate/Up 的"打包"

[Qwen3Attention](../nanovllm/models/qwen3.py) 的 `qkv_proj` 用 [QKVParallelLinear](../nanovllm/layers/linear.py)：

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden, head_size, total_num_heads, total_num_kv_heads):
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden, output_size)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # loaded_shard_id ∈ {"q", "k", "v"}
        # 按 q/k/v 计算 shard_offset / shard_size
        # 从 HF 单独的 q_proj/k_proj/v_proj 中切片填入合并后的 qkv 中
        ...
```

把 q/k/v **沿输出方向拼成一块**，做一次大 GEMM 比做三次小 GEMM 显著更快（GPU 喜欢大矩阵）。

类似地，MLP 的 `gate_up_proj` 用 [MergedColumnParallelLinear](../nanovllm/layers/linear.py) 把 `gate_proj` 与 `up_proj` 合并。

## 6.5 VocabParallelEmbedding 与 ParallelLMHead

[embed_head.py](../nanovllm/layers/embed_head.py) 把 vocab 维切片：

```python
class VocabParallelEmbedding(nn.Module):
    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)    # 越界 token 临时置 0
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                # 越界位置的 embedding 清零
            dist.all_reduce(y)                       # 各卡相加 → 完整 embedding
        return y
```

巧妙之处：**用 mask 把越界 id 的查询结果清零，all-reduce 之后只有真正负责这个 vocab 区间的卡贡献了非零值**。

`ParallelLMHead.forward` 在 prefill 时只取最后位置（参见上一章），并在 TP 时 gather logits 到 rank 0：

```python
all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
dist.gather(logits, all_logits, 0)
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

只有 rank 0 拿到完整 logits 用于采样，其他卡不参与。

## 6.6 多进程协同：SharedMemory + Event

[ModelRunner.__init__](../nanovllm/engine/model_runner.py)：

```python
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()                                  # 子进程进入消息循环
```

控制流通信用 1 MB 共享内存 + `multiprocessing.Event`：

```python
def write_shm(self, method_name, *args):              # 仅 rank 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()                                   # 通知所有 worker

def read_shm(self):                                   # 仅 worker
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

[call()](../nanovllm/engine/model_runner.py) 是统一入口：

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)            # 广播给 worker
    method = getattr(self, method_name, None)
    return method(*args)
```

worker 端的主循环：

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

> 这种"共享内存广播控制流 + NCCL 处理张量"的组合是真实 vLLM 早期版本的做法（V1 引擎已经重写）。

## 6.7 数据流图（TP=2）

```
              ┌─────────────────┐
   input_ids  │   prepare_*     │   主进程独有
              └────────┬────────┘
                       │ pickle 进 SharedMemory，set Event
              ┌────────▼────────┐
              │  embed_tokens   │  Vocab 切 → 输出走 all-reduce
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  QKV (Column)   │  W 切输出，无通信
              ├─────────────────┤
              │  Attention      │  各卡 num_heads/tp_size 个头
              ├─────────────────┤
              │  O (Row)        │  W 切输入，all-reduce ◄── 1 次
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Gate/Up (Col)  │  无通信
              │  SiLU & Mul     │
              │  Down (Row)     │  all-reduce ◄── 1 次
              └────────┬────────┘
                       │
              … N 层 …
                       │
              ┌────────▼────────┐
              │  LM Head        │  TP 时 gather logits 到 rank 0
              └─────────────────┘
```

每个 transformer layer：**2 次 all-reduce**。

## 6.8 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| 控制流：SharedMemory + Event | Ray RPC（默认）或 ZMQ |
| `Sequence.__getstate__` 手写精简 | `SequenceData` 也有类似优化 |
| 仅 TP，没有 PP | TP + PP + EP（专家并行） |
| LM Head 用 gather | 也支持 all-gather + 本地 argmax |

## 6.9 动手任务

1. 把 `tensor_parallel_size` 改成 2（需要 2 卡 GPU），观察启动多进程的过程
2. 把 `RowParallelLinear.forward` 中的 `dist.all_reduce(y)` 注释掉，跑测试，观察输出错误
3. 思考：如果 `num_attention_heads = 28`，`tp_size = 8`，能跑吗？
   *答：不能。`Qwen3Attention.__init__` 里有 `assert total_num_heads % tp_size == 0`，因为不能把头数切不整。*
4. 思考：为什么 `VocabParallelEmbedding` 需要 all-reduce 而 `ColumnParallelLinear` 不需要？
   *答：embedding 输出 `[N, hidden]` 需要每张卡都拿到完整结果给后续层；而 `ColumnParallelLinear` 的下游通常是 `RowParallelLinear`，分片输出正好作为切片输入，无需提前合并。*

---

上一章 ← [05 · Prefill 与 Decode](./05-prefill-vs-decode.md) | 下一章 → [07 · CUDA Graph](./07-cuda-graph.md)
