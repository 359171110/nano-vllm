# 01 · 整体架构与请求生命周期

## 学习目标

读完本章，你应该能回答：

- nano-vllm 把"提供推理服务"这件事拆成了哪几层？各自负责什么？
- 一个 prompt 从输入到输出经历了哪些步骤？
- `LLMEngine.step()` 到底做了什么？为什么是引擎的核心？

## 涉及源码

- [nanovllm/llm.py](../nanovllm/llm.py)
- [nanovllm/engine/llm_engine.py](../nanovllm/engine/llm_engine.py)
- [nanovllm/engine/sequence.py](../nanovllm/engine/sequence.py)
- [nanovllm/config.py](../nanovllm/config.py)

## 1.1 三层抽象

vLLM（以及 nano-vllm）的核心不是模型，而是 **请求调度 + KV Cache 管理 + 高效算子** 三件事的协同。它们被分成三层：

```
LLM (用户接口)                     nanovllm/llm.py
  └── LLMEngine (引擎/驱动)         nanovllm/engine/llm_engine.py
        ├── Scheduler              nanovllm/engine/scheduler.py
        │     └── BlockManager      nanovllm/engine/block_manager.py
        └── ModelRunner             nanovllm/engine/model_runner.py
```

| 层 | 职责 | 关键问题 |
|---|---|---|
| `LLM` / `LLMEngine` | 接收请求、跑主循环、返回结果 | 什么时候停？ |
| `Scheduler` | 每一步选出"算谁、算多少 token" | 显存够不够？要不要抢占？ |
| `BlockManager` | 管理 KV Cache 物理块的分配/回收/复用 | 块从哪来？可不可以共享？ |
| `ModelRunner` | 准备张量、跑前向、采样 | 怎么把变长 batch 喂给 FlashAttention？ |

> 真实 vLLM 把每一层都做得更复杂，但层次完全相同。

## 1.2 请求的生命周期

```
add_request(prompt)
        │
        ▼
   Sequence(WAITING)  ──►  Scheduler.waiting (deque)
                                  │
                                  ▼  (每一步循环)
                          Scheduler.schedule()
                                  │
                ┌─────────────────┼─────────────────┐
                ▼ (prefill)                       ▼ (decode)
        还有 prompt 没算的序列              已经在 RUNNING 的序列
                │                                 │
                └─────────────►  ModelRunner.run  ◄──┘
                                  │
                                  ▼
                         Scheduler.postprocess()
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                                 ▼
            还没结束                       EOS / max_tokens 满
            （回到 running）                状态置 FINISHED, 释放 KV
```

## 1.3 LLMEngine 精读

入口约 90 行：[llm_engine.py](../nanovllm/engine/llm_engine.py)。

### 1.3.1 初始化：为什么 `spawn` 多进程？

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
self.model_runner = ModelRunner(config, 0, self.events)
```

- 主进程（`rank=0`）自己也跑一份 `ModelRunner`
- 当 `tensor_parallel_size > 1` 时，再 spawn 出 `tp-1` 个 worker 进程
- 计算用 NCCL，控制流用 `multiprocessing.Event + SharedMemory`

> **为什么用 spawn 而不是 fork？**
> CUDA 上下文不能跨 fork，会死锁。`spawn` 让子进程从零启动 Python 解释器，干净彻底。

#### Q: spawn 是啥？fork 又是啥？为什么 CUDA 不能跨 fork？

这是 Python 多进程的两种"启动方式"，差别在于子进程的初始状态。

**fork（POSIX 经典做法）**

- 系统调用：把当前进程的整个内存空间复制一份（写时复制 COW）
- 子进程从 `fork()` 调用的下一行继续执行，父进程的所有变量、已加载的库、打开的文件描述符、已分配的内存全部「继承」过来
- Linux / macOS 上 Python 早期的默认方式
- 优点：启动极快（COW，几乎零拷贝），可共享父进程已加载的大对象（例如已经 import 好的 torch、已经加载到内存的权重）
- 缺点：复制了一切**不该复制的状态**——包括 CUDA 上下文、后台线程、互斥锁

**spawn（跨平台稳妥做法）**

- 启动一个**全新的** Python 解释器进程
- 用 pickle 把目标函数 + 参数序列化传过去，子进程重新 `import` 所需模块，从零执行
- Windows 唯一支持的方式；Linux + CUDA 场景下 PyTorch 强烈推荐
- 优点：干净，不继承任何父进程状态
- 缺点：启动慢（重新 import torch 可能要几秒，权重也要重新加载）

**为什么 CUDA 不能跨 fork**

CUDA driver 在父进程里维护着一堆**进程级**状态：

1. **CUDA context**：每个进程持有 GPU 的「会话」，含显存映射表、stream、命令队列
2. **后台线程**：CUDA runtime 启动了多个 worker 线程做命令提交、内存管理
3. **互斥锁 / 条件变量**：保护上述状态

`fork()` 只复制**调用 fork 的那一个线程**，其他线程「凭空消失」，但锁、内存映射被原封不动地复制进子进程。后果：

- 子进程里出现「已上锁但没有持有者」的死锁
- CUDA driver 检测到 PID 变化，下次调用 CUDA API 直接报 `CUDA error: initialization error`
- 即使勉强能调用，父子进程对同一个 context 并发操作是 UB（未定义行为）

所以 PyTorch 在用 CUDA 时强制要求 spawn：

```python
ctx = mp.get_context("spawn")    # ← 必须 spawn
process = ctx.Process(target=ModelRunner, args=...)
```

**对比一图**

```
fork:
  父进程 [变量 + CUDA context + 后台线程 + 互斥锁]
     │ fork()  ── 只复制当前线程
     ▼
  子进程 [继承一切，但只有 1 个线程活着，CUDA 状态错乱]
         └── 调用 CUDA API → 死锁 / 报错

spawn:
  父进程 [略]
     │ pickle(target, args) → 启动新解释器
     ▼
  子进程 [全新进程，重新 import torch，独立初始化 CUDA]
         └── 干净启动，独立 CUDA context
```

> 顺带一提：Python 还有第三种 `forkserver` 方式（先 fork 一个干净的 server 进程，再由它 fork worker），不过 CUDA 场景下大家普遍直接用 spawn。

#### Q: spawn 多进程是不是只在 `tensor_parallel_size > 1` 时才用？

**是的。** 看循环边界：

```python
for i in range(1, config.tensor_parallel_size):
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
self.model_runner = ModelRunner(config, 0, self.events)
```

- `tensor_parallel_size = 1`：`range(1, 1)` 是空的，**一个子进程都不会创建**。整个引擎运行在主进程里，包括 rank=0 的 `ModelRunner`。
- `tensor_parallel_size = N (> 1)`：spawn 出 `N - 1` 个 worker。主进程是 rank 0，子进程是 rank 1 到 rank N-1。加上主进程刚好 N 个 rank，与 GPU 数一一对应。

举个例子：

| `tensor_parallel_size` | 主进程角色 | spawn 出的子进程 | 总 GPU 数 |
|---|---|---|---|
| 1 | rank 0 | 无 | 1 |
| 2 | rank 0 | rank 1 | 2 |
| 4 | rank 0 | rank 1 / 2 / 3 | 4 |
| 8 | rank 0 | rank 1 … 7 | 8 |

所以如果你跑 `example.py`（默认 `tensor_parallel_size=1`），不会看到任何额外进程，也不会用到 `SharedMemory + Event` 那套控制流；NCCL 这些跨卡通信代码也只在 TP > 1 时才走到。

> 但注意：即使 `tp_size=1`，[`ModelRunner.__init__`](../nanovllm/engine/model_runner.py) 里 **`dist.init_process_group("nccl", ...)` 仍然会调用**——以 world_size=1 初始化。这是为了让 `linear.py`、`embed_head.py` 里 `dist.get_world_size()` / `dist.get_rank()` 这些调用能统一生效，不用在每个算子里写 `if tp_size > 1:` 分支。

### 1.3.2 主循环：[generate()](../nanovllm/engine/llm_engine.py)

```python
while not self.is_finished():
    output, num_tokens = self.step()
    # 统计 prefill / decode 吞吐
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids
```

把所有请求一次性 `add_request` 进去，然后**反复 `step` 直到 waiting + running 都为空**。

### 1.3.3 一步：[step()](../nanovllm/engine/llm_engine.py)

仅 7 行，却是整个引擎的灵魂：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

三句话：
1. **调度**：决定本步算哪些序列、走 prefill 还是 decode
2. **执行**：跑前向 + 采样
3. **收尾**：写回 token、检查终止条件、回收 KV 块

> `num_tokens` 用正负号区分 prefill / decode，是个偷懒小技巧；真实 vLLM 用 dataclass 区分。

## 1.4 Sequence：状态机的最小单位

[Sequence](../nanovllm/engine/sequence.py) 是单个请求在引擎内部的"对象化身"，状态只有 3 个：

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 在 scheduler.waiting 里
    RUNNING = auto()    # 在 scheduler.running 里，每步生成 1 token
    FINISHED = auto()   # 已结束，等待回收
```

关键字段（**必背**）：

| 字段 | 含义 |
|---|---|
| `token_ids` | 完整 token 序列（prompt + 已生成） |
| `num_prompt_tokens` | prompt 长度 |
| `num_cached_tokens` | KV Cache 中已经"算好"的前缀长度 |
| `num_scheduled_tokens` | 本步计划要算的 token 数（prefill 可变，decode 恒为 1） |
| `block_table` | 该序列占用的 KV 物理块编号列表 |
| `is_prefill` | 当前是否处于 prefill 阶段 |

> `num_cached_tokens` 是理解前缀缓存与分块预填充的关键。

## 1.5 Config：贯穿教程的几个阈值

[Config](../nanovllm/config.py) 中的关键参数：

| 字段 | 默认 | 作用 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | 单步最多算多少 token（prefill 上限） |
| `max_num_seqs` | 512 | 单步最多多少条序列（decode batch 上限） |
| `max_model_len` | 4096 | 单序列最大上下文长度 |
| `gpu_memory_utilization` | 0.9 | 给 KV Cache 留多少显存 |
| `kvcache_block_size` | 256 | 每个 KV 物理块的 token 数 |
| `tensor_parallel_size` | 1 | TP 卡数 |
| `enforce_eager` | False | 是否禁用 CUDA Graph |

## 1.6 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| `LLM` 直接继承 `LLMEngine` | `LLM` 包装 `LLMEngine`，多了批量提交、流式输出 |
| `step()` 单线程同步 | `AsyncLLMEngine` 异步化，服务模式下使用 |
| Worker 用 spawn + Event + SharedMemory | 默认 Ray，旧版本也支持 multiprocessing |
| Sequence 状态只有 3 个 | 多 SWAPPED、Beam Search 状态等 |

## 1.7 动手任务

1. 在 [step()](../nanovllm/engine/llm_engine.py) 里 `print(is_prefill, len(seqs), [s.num_scheduled_tokens for s in seqs])`，跑一次 `example.py`，观察序列在 prefill 与 decode 间切换的节奏
2. 把 `max_num_batched_tokens` 改成 1024，跑长 prompt，看是否被切成多步 prefill（为下一章做铺垫）
3. 阅读 [Sequence.\_\_getstate\_\_/\_\_setstate\_\_](../nanovllm/engine/sequence.py)，理解为什么传给 worker 的序列在 prefill / decode 阶段序列化的内容不同（提示：网络带宽优化）

---

下一章 → [02 · PagedAttention 与 KV Cache 块管理](./02-paged-attention.md)
