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

CUDA Graph 把整个前向“录制”成一张图，重放时跳过所有 host 端开销，只让 GPU 按预录的 kernel 序列执行。

#### Q: 原来到底慢在哪？用了 CUDA Graph 为什么能快？

**先理解 GPU 的工作模式**

GPU 不能自己决定算什么，它靠 CPU 下达指令（类似"老板下任务，工人干活"）：

```
CPU (老板)                              GPU (工人)
│                                         │
├─ 第 1 个算子: RMSNorm               │
│   [Python 调度 ~5µs]                   │
│   [CUDA launch ~3µs]                   │
│   ──── 发送指令 ───▶                 ├─ 执行 RMSNorm [~2µs]
├─ 第 2 个算子: QKV Linear             │
│   [Python 调度 ~5µs]                   │
│   [CUDA launch ~3µs]                   │
│   ──── 发送指令 ───▶                 ├─ 执行 GEMM [~10µs]
├─ 第 3 个算子: RoPE                   │
│   ...                                   ...
```

看到问题了吗？**CPU 每发一条指令就要花 ~8µs（Python解释 + CUDA launch），但 GPU 实际干活可能只要 2~10µs。**

decode 时 batch 很小（每条序列只算 1 个 token），每个 kernel 的计算量微乎其微，但 CPU 端的“行政开销”不变。一个 28 层的模型，每层约 10 个算子 = 280 个算子：

```
实际计算: 280 × ~5µs (GPU) = ~1.4ms
CPU 开销: 280 × ~8µs        = ~2.2ms
总耗时:                        ~3.6ms
GPU 利用率: 1.4/3.6           = 39%  ← 超过一半时间 GPU 在等 CPU
```

这就是“慢”的本质：**GPU 大部分时间在等 CPU 下达指令，而不是在计算。**

**CUDA Graph 怎么解决**

思路：既然每次 decode 执行的算子序列都一样（同样的模型、同样的形状），为什么每次都要 CPU 重新发 280 条指令？“录”一遍，以后直接“放”不就行了？

```
录制阶段（启动时执行一次）:
  CPU 正常发送 280 个算子指令，CUDA 把它们录成一张图

重放阶段（每步 decode 时）:
  CPU 只发 1 条指令: graph.replay()
  GPU 收到后按图里录好的顺序自动执行全部 280 个 kernel
```

用回上面的数字：

```
不用 Graph:
  CPU 开销: 280 × 8µs = 2.2ms
  GPU 计算: ~1.4ms
  总耗时: ~3.6ms

用 Graph:
  CPU 开销: 1 × ~10µs = 0.01ms  (replay 只是一次调用)
  GPU 计算: ~1.4ms              (和之前一样)
  总耗时: ~1.4ms

加速比: 3.6 / 1.4 = 2.6×
```

**一句话总结**

> 慢的原因：CPU 每个算子都要“Python 调度 + CUDA launch”，decode 时算量小但算子多，“行政开销”比实际干活还耗时。
> 快的原因：录制一次，以后每步只发 1 条 `replay()` 指令，CPU 开销从 280 次压缩到 1 次。

**为什么 prefill 不用 CUDA Graph？**

prefill 时每个 token 的计算量很大（长序列的大 GEMM），GPU 计算时间 >> CPU 发送指令时间，host 开销占比忽略不计，而且 prefill 每次的序列长度都不一样，桃化代价太大。只有 decode（batch 小、算量小、算子多）才是 CUDA Graph 的最佳场景。

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
# ===== 第一部分：创建固定 buffer =====
# 这些张量的地址一旦分配就不会变，整个 CUDA Graph 生命期都用这几个 buffer。
# 形状用 max_bs（最大 batch size）分配，每次重放前拷入实际数据。

input_ids    = torch.zeros(max_bs, dtype=torch.int64)       # 每条序列的输入 token
positions    = torch.zeros(max_bs, dtype=torch.int64)       # 每条序列的位置编号
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)       # 每个 token 的 KV 写入位置
context_lens = torch.zeros(max_bs, dtype=torch.int32)       # 每条序列的总长度（FlashAttn 需要）
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # 每条序列的页表
outputs      = torch.zeros(max_bs, hidden_size)             # 模型输出的 hidden states

# ===== 第二部分：为每个桶 batch size 录制一张图 =====

for bs in reversed(self.graph_bs):       # 从大到小遍历（如 512, 496, ..., 8, 4, 2, 1）
    graph = torch.cuda.CUDAGraph()       # 创建一个空的 graph 对象

    # 设置 attention 层需要的全局上下文（decode 模式）
    # 这里的 [:bs] 是只取前 bs 行，模拟“当前 batch 有 bs 条序列”
    set_context(False,                   # is_prefill=False，因为 graph 只用于 decode
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs])

    # --- warmup: 让 PyTorch 走一遍前向，触发所有 lazy 初始化 ---
    # 第一次调用时 PyTorch/cuBLAS 会做很多一次性的事：
    #   - 选择最优 GEMM kernel
    #   - 分配 cuBLAS workspace
    #   - 编译 torch.compile 的 kernel
    # 这些都不能被录进 graph，所以先跑一遍“排掉”它们
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

    # --- capture: 正式录制 ---
    # with 块里面的所有 CUDA 操作会被录制到 graph 中
    # self.graph_pool: 所有图共享的中间显存池（省显存）
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

    # 第一张图创建 pool，后续的图复用它
    if self.graph_pool is None:
        self.graph_pool = graph.pool()

    # 把录好的图存起来，以 batch size 为 key
    self.graphs[bs] = graph

# ===== 第三部分：记下固定 buffer 的引用 =====
# 重放时要往这些 buffer 里拷数据，所以要保存它们的引用
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

#### Q: warmup 里“选择最优 GEMM kernel / 分配 workspace / 编译 kernel”都是什么？为什么不能录进 Graph？

**三件事分别是：**

| 一次性操作 | 具体在干什么 | 为什么不能录 |
|---|---|---|
| cuBLAS autotuning | GPU 做 GEMM 有几十种实现（不同 tile、不同访存模式）。第一次调用时 cuBLAS 会试几种，选最快的记住 | 会录到额外的 benchmark kernel，replay 时会重复执行它们 |
| 分配 workspace | cuBLAS 需要临时显存存中间结果，第一次才分配（lazy） | `cudaMalloc` 会改变 tensor 地址，但 Graph 要求地址固定 |
| 编译 kernel | `@torch.compile` / Triton 第一次调用时把 Python 编译成 CUDA kernel | 编译是纯 host 端行为，不产生 GPU 操作；编译前 kernel 函数指针还不存在 |

**核心原因只有一个：**

> CUDA Graph 要求：录制时看到的 **kernel 序列、tensor 地址、形状**，在每次 replay 时都完全一样。

上面三件事都违反了这个规则：autotuning 会混入多余 kernel，malloc 会改地址，编译前 kernel 不存在。

**一个类比：**

你要录一段烹饪视频：
- 试几把刀看哪把顺手（autotuning）— 不录进正片
- 去超市买菜（分配 workspace）— 不录进正片
- 把菜谱从英文翻成中文（编译 kernel）— 不录进正片

warmup 就是“开拍前准备”，capture 才是“正式开拍”——只录纯净的、可重复的做菜步骤。

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

#### Q: `if is_prefill or self.enforce_eager or input_ids.size(0) > 512` 这行是“不满足用图的条件就退化为常规前向传播”吧？

**是的。** 这行是 CUDA Graph 的“守门员”，三个条件任一成立就走普通 eager 前向：

| 条件 | 为什么不能用 Graph |
|---|---|
| `is_prefill` | prefill 的 token 数每次不同（可能几十到几千），无法桶化；而且 prefill 计算量大，CPU 开销占比忽略不计，用不着图 |
| `self.enforce_eager` | 用户显式指定 `enforce_eager=True`，常用于 debug（图模式下报错难定位） |
| `input_ids.size(0) > 512` | batch 超过了录制时的最大 bs，没有对应的图可用 |

“退化”的实现就是一行：

```python
return self.model.compute_logits(self.model(input_ids, positions))
```

直接调用模型的 `forward`，走正常的 PyTorch eager 执行（每个算子由 CPU 逐个发射）。功能完全相同，只是没有 CUDA Graph 的加速。

而 else 分支才是走图的路径：拷数据到固定 buffer → `graph.replay()` → 取结果。

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
