# 04 · 调度器：Continuous Batching + Chunked Prefill + Preempt

## 学习目标

- 为什么 prefill 与 decode 不能混在一起算？
- chunked prefill 怎么把超长 prompt 分多步喂？
- 显存不够时谁被牺牲、怎么恢复？

## 涉及源码

- [nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py)

## 4.1 调度的两条路径

[Scheduler.schedule()](../nanovllm/engine/scheduler.py) 的骨架：

```python
def schedule(self):
    scheduled_seqs = []
    num_batched_tokens = 0

    # 路径 1：优先 prefill
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        ...
    if scheduled_seqs:
        return scheduled_seqs, True

    # 路径 2：没有 prefill 任务才做 decode
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        ...
    return scheduled_seqs, False
```

**优先级：有 waiting 就先 prefill，否则 decode**。原因：

| 阶段 | 计算特性 | 喜欢的 batch 模式 |
|---|---|---|
| prefill | compute-bound（长 seq、密集计算） | 单次塞满 token 预算即可 |
| decode | memory-bound（每步 1 token、纯读 KV） | batch 越大越省 |

两者混合会让 attention kernel 的 batched mask 极其复杂、effective FLOPs 都被浪费。

> 真实 vLLM 在新版本中默认开启 chunked prefill，把 prefill chunk 与 decode 同步，但 nano-vllm 的写法是早期 vLLM 的经典做法，更易理解。

## 4.2 Prefill 路径详解

### 4.2.1 算力预算：max_num_batched_tokens

```python
remaining = self.max_num_batched_tokens - num_batched_tokens
if remaining == 0:
    break
```

整个 batch 加起来不能超过 `max_num_batched_tokens`（默认 16384）。这是单步算力预算。

### 4.2.2 块预算：can_allocate

```python
if not seq.block_table:                          # 全新的序列
    num_cached_blocks = self.block_manager.can_allocate(seq)
    if num_cached_blocks == -1:                  # 块不够
        break
    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
else:                                            # 已经分块过（例如被 chunk / 抢占恢复）
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

### 4.2.3 Chunked Prefill：让超长 prompt 分多步进

```python
if remaining < num_tokens and scheduled_seqs:
    break                                        # 后面的序列等下一步
seq.num_scheduled_tokens = min(num_tokens, remaining)
num_batched_tokens += seq.num_scheduled_tokens
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING          # prefill 完成，转 RUNNING
    self.waiting.popleft()
    self.running.append(seq)
scheduled_seqs.append(seq)
```

注意 `and scheduled_seqs` 这个条件：**只有 batch 里已经有别人时，才跳过本序列**。如果当前 batch 是空的，即使本序列要算的 token 数超过 `remaining`，也允许它"独占"本步算一段——这就是 chunked prefill 的入口。

序列若没算完（`num_cached_tokens + num_scheduled_tokens < num_tokens`），依然停留在 waiting 队列；算完才转 RUNNING。

### 4.2.4 一个例子

`max_num_batched_tokens = 1000`，三条序列要 prefill，长度 800 / 600 / 400：

| Step | scheduled | 动作 |
|---|---|---|
| 1 | seq1(800) | seq1 完整 prefill；seq2 不进（800+600 > 1000） |
| 2 | seq2(600), seq3(400) | seq2 + seq3 = 1000，正好填满 |
| 3 | (decode) | 全部进入 decode |

如果 seq1 的长度变成 1500：

| Step | scheduled | 动作 |
|---|---|---|
| 1 | seq1(1000/1500) | 单独占用，只算前 1000 |
| 2 | seq1(500/1500) | 续算 500，完成 prefill，转 RUNNING |
| 3 | (decode 或新进入) | … |

## 4.3 Decode 路径详解

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())   # 牺牲队尾
        else:
            self.preempt(seq)                  # 没人可牺牲，自己被牺牲
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

### 4.3.1 can_append：为什么只查 1 块？

decode 每步只生成 1 个 token，只有当当前块刚好被填满时才需要新块：

```python
def can_append(self, seq):
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` 表示新加的这个 token 是新块的第一个，需要分配一块新块。

### 4.3.2 抢占（Preempt）

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

被牺牲的序列：
- 状态回到 WAITING（**队首**，优先恢复）
- KV 块全部释放
- 标记 `is_prefill=True`，下一轮 prefill 重新算（recompute）

> 真实 vLLM 还有 swap 模式：把 KV 拷到 CPU 内存，恢复时拷回。nano-vllm 选 recompute，因为代码简单且对短序列其实更快。

### 4.3.3 牺牲谁？

```python
self.preempt(self.running.pop())   # 抢的是队尾
```

队尾通常是最晚加入 RUNNING 的——FIFO 公平。

## 4.4 postprocess：闭合循环

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)              # ① 给新填满的块加哈希
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                                     # 还在分块 prefill，不接收 token
        seq.append_token(token_id)                       # ② 写入新 token
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

注意 chunked prefill 中间步骤虽然返回了 token_id（最后一个位置的输出），但 **不会被采纳**——因为整个 prompt 还没 prefill 完。

## 4.5 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| prefill / decode 严格分步 | 默认开启 chunked prefill，prefill chunk 与 decode 混跑 |
| 抢占只能 recompute | 支持 swap-to-CPU |
| 单一队列 `deque` | 多优先级队列（prefill / running / swapped） |
| 牺牲队尾 | 多种 policy：FCFS / priority |
| 不区分 LoRA | 同一 LoRA 的请求会被分组调度 |

## 4.6 动手任务

1. 把 `max_num_seqs` 改到 2，跑 4 条长 prompt，观察是否触发抢占
2. 在 `preempt` 里 `print(seq.seq_id)`，看 FIFO 抢占顺序
3. 思考：为什么 chunked prefill 只允许 batch 中的"第一条"序列 chunk？
   *答：简化实现。允许多条 chunk 时，需要在算力预算内为每条单独裁剪长度，状态机会复杂得多。*
4. 思考：被抢占的序列，恢复时它的前缀缓存能命中吗？
   *答：能。释放出去的块只要还在 free 队列里、`hash_to_block_id` 还指向它，下一轮 `can_allocate` 就会命中——这就是 recompute 模式的真实代价比想象中小的原因。*

---

上一章 ← [03 · 前缀缓存](./03-prefix-caching.md) | 下一章 → [05 · Prefill 与 Decode 两条数据通路](./05-prefill-vs-decode.md)
