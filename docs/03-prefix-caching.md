# 03 · 前缀缓存：把 KV 变成内容寻址缓存

## 学习目标

- 多个序列共享相同前缀时，KV 怎么复用？
- 哈希怎么算才能精确匹配前缀？
- 引用计数 + 双队列回收为什么能"零成本"复活已释放的块？

## 涉及源码

- [nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)

## 3.1 设计动机

考虑这些场景：
- 系统 prompt 数千 token，每次对话都要 prefill 一次
- 多用户的 RAG 拼接相同的检索结果
- Beam Search 多个候选共享同一前缀

如果能让相同前缀只 prefill 一次、KV 直接复用，prefill 延迟可以从秒级降到毫秒级。

vLLM 的方案：**基于哈希的内容寻址 KV Cache**。

## 3.2 链式哈希：让"前缀"可识别

每个 KV 块的哈希 = `hash(本块 token_ids ++ 前一块的哈希)`：

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

链式带来的关键性质：

> **两条序列的第 k 块哈希相等 ⟺ 前 k 块所有 token 都相等。**

这使得“前缀匹配”等价于“找最长哈希相同前缀”，无需逐 token 比较。

#### Q: 链式哈希怎么设计的？为什么要“链”？

分三个层次看。

**“普通哈希”要怎么做（不链的​话）**

最朴素的想法：给每个块 i 的哈希就算 `hash(block_i 的 token_ids)`——比如块 2 的内容是 `[512, 1034, 998, 3721]`，哈希 = `xxh64([512, 1034, 998, 3721])`。

问题在哪？假设两条不同的序列 A、B：
- A：块 0 = `[hello]`，块 1 = `[world]`，块 2 = `[foo]`
- B：块 0 = `[DIFFERENT]`，块 1 = `[world]`，块 2 = `[foo]`

A 的块 2 哈希 == B 的块 2 哈希（都是 `hash([foo])`）。但它们前缀不同！如果复用了 A 的 KV块 2 给 B，答案就错了——因为 attention 要看“从头到当前位置”的全部 K/V，前面不一样，后面的 attention 结果就不一样。

> “块 i 的 KV 可以复用” 的充要条件是 **前 i 块的全部 token 也都相同**。

**“链式哈希”怎么解决这个问题**

思路非常简单：**块 i 的哈希输入 = 块 i 的 token_ids + 块 i-1 的哈希**。

```
hash_0 = xxh64(block_0_tokens)                       # 第 0 块：只算自己
hash_1 = xxh64(hash_0 ++ block_1_tokens)             # 第 1 块：带上前一块的哈希
hash_2 = xxh64(hash_1 ++ block_2_tokens)             # 第 2 块：带上前两块的信息
...
```

用代码写就是：

```python
def compute_hash(cls, token_ids, prefix=-1):  # prefix 就是上一块的 hash
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 把上一块的 hash 作为输入的一部分
    h.update(np.array(token_ids).tobytes())      # 再加上本块的 token
    return h.intdigest()
```

于是：
- `hash_2` 里「编码」了块 0、1、2 全部的 token 信息
- **只有前 3 块完全一样的序列，算出来的 hash_2 才相同**
- 前面任何一个 token 不同，后续所有块的哈希都不同（雪崩效应）

**回到上面的例子**

```
A: block0=[hello], block1=[world], block2=[foo]
B: block0=[DIFFERENT], block1=[world], block2=[foo]
```

- A.hash_0 = xxh64([hello])，B.hash_0 = xxh64([DIFFERENT]) → ≠
- A.hash_1 = xxh64(A.hash_0 ++ [world])，B.hash_1 = xxh64(B.hash_0 ++ [world]) → ≠（前缀不同）
- A.hash_2 = xxh64(A.hash_1 ++ [foo])，B.hash_2 = xxh64(B.hash_1 ++ [foo]) → ≠（前缀不同）

→ B 的任何块都无法复用 A 的 KV 缓存。安全。

而如果 C 和 A 的前缀相同：

```
C: block0=[hello], block1=[world], block2=[bar]
```

- C.hash_0 = A.hash_0 → ✔ 复用块 0 的 KV
- C.hash_1 = A.hash_1 → ✔ 复用块 1 的 KV
- C.hash_2 = xxh64(A.hash_1 ++ [bar]) ≠ A.hash_2 → ✘ 不能复用，重新算

→ 正确地复用了前 2 块，节省了 prefill 时间。

**一句话总结**

> 链式哈希 = 「本块内容 + 前一块哈希」的嵌套折叠。它把“前 k 块 token 全部相同”这件事压缩成“第 k 块哈希相等”一个比较，让前缀匹配变成 O(num_blocks) 表查找，而不是 O(num_tokens) 的逻 token 比较。

## 3.3 hash_to_block_id 表的维护

[hash_blocks()](../nanovllm/engine/block_manager.py) 在每次"块被填满"后被调用（由 `Scheduler.postprocess` 触发）：

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // block_size
    end   = (seq.num_cached_tokens + seq.num_scheduled_tokens) // block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

注意：**只哈希填满的块**。半满的块永远不入表，因为下一步还会写入、token_ids 会变。

#### Q: hash_blocks 的代码举个例子走一遍？

设定：`block_size = 4`，序列 A 总长 12 个 token，`block_table = [7, 9, 3]`（占了 3 个物理块）。

假设本步是序列刚刚完成完整 prefill（一次性填了 12 个 token）：
- `num_cached_tokens = 0`（prefill 前还没缓存）
- `num_scheduled_tokens = 12`（本步算了 12 个）

**第 1 步：计算 start 和 end**

```python
start = num_cached_tokens // block_size   = 0 // 4 = 0
end   = (0 + 12) // block_size            = 12 // 4 = 3
```

意思：块 0、1、2 在本步被填满了，需要给它们算哈希。

> 为什么用整除？因为只有**填满**的块才入表。如果 `num_scheduled_tokens = 10`，那 `end = 10//4 = 2`，只给块 0、1 算（块 2 只填了 2 个 token，还没满）。

**第 2 步：拿到前一块的哈希作为链式起点**

```python
h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
# start=0，所以 h = -1（没有前一块）
```

如果是 chunked prefill 的第二次（比如 `start=2`），就会拿块 1 的哈希作为链式起点，继续往下算。

**第 3 步：逐块计算哈希并入表**

```
i=0:
  block    = blocks[block_table[0]] = blocks[7]   # 物理块 7
  token_ids= seq.block(0) = [t0, t1, t2, t3]
  h        = compute_hash([t0,t1,t2,t3], prefix=-1)
           = xxh64([t0,t1,t2,t3])                 # 第一块，没有前缀
  blocks[7].hash      = h
  blocks[7].token_ids = [t0,t1,t2,t3]
  hash_to_block_id[h] = 7                          # 入表！

i=1:
  block    = blocks[block_table[1]] = blocks[9]
  token_ids= seq.block(1) = [t4, t5, t6, t7]
  h        = compute_hash([t4,t5,t6,t7], prefix=上一步的h)
           = xxh64(上一步h ++ [t4,t5,t6,t7])    # 链式！
  blocks[9].hash      = h
  blocks[9].token_ids = [t4,t5,t6,t7]
  hash_to_block_id[h] = 9

i=2:
  block    = blocks[block_table[2]] = blocks[3]
  token_ids= seq.block(2) = [t8, t9, t10, t11]
  h        = compute_hash([t8,t9,t10,t11], prefix=上一步的h)
           = xxh64(上一步h ++ [t8,t9,t10,t11])  # 链上前两块的信息
  blocks[3].hash      = h
  blocks[3].token_ids = [t8,t9,t10,t11]
  hash_to_block_id[h] = 3
```

**执行后状态**

| 物理块 | 保存的 hash | 保存的 token_ids | 含义 |
|---|---|---|---|
| 7 | h0 | [t0,t1,t2,t3] | "前 4 个 token 是这些"的证据 |
| 9 | h1 | [t4,t5,t6,t7] | "前 8 个 token 是这些"的证据 |
| 3 | h2 | [t8,t9,t10,t11] | "前 12 个 token 是这些"的证据 |

之后如果有新序列的前缀与 A 相同，`can_allocate` 就能通过 `hash_to_block_id` 查到 7、9、3，直接复用它们的 KV，跳过 prefill。

**Chunked Prefill 场景补充**

如果同一个序列的 prefill 被分成两步（第一步算 8 token，第二步算 4 token）：

```
第一步 postprocess:
  num_cached_tokens=0, num_scheduled_tokens=8
  start=0, end=2 → 给块 0、1 算哈希入表
  num_cached_tokens 更新为 8

第二步 postprocess:
  num_cached_tokens=8, num_scheduled_tokens=4
  start=8//4=2, end=(8+4)//4=3 → 只给块 2 算哈希
  h = blocks[block_table[1]].hash   # 拿块 1 的 hash 作为链式起点
  给块 2 计算并入表
```

结果与一次性 prefill 完全一致——这就是 `h = blocks[block_table[start-1]].hash if start > 0 else -1` 这行的意义：**从上一次断点接着链下去**。

#### Q: 判断复用的逻辑是不是“从第一块开始逐块算哈希、查表、直到某一块没命中为止”？

**思路完全正确，但有一个关键细节：粒度是「块」而不是「单个 token」。**

完整流程：

```
新序列入场，prompt = [t0, t1, t2, ..., t99]  (100 个 token)
block_size = 4 → 分成 25 块

第 0 块: token_ids = [t0, t1, t2, t3]
  h = compute_hash([t0,t1,t2,t3], prefix=-1)    # 算哈希
  查表: hash_to_block_id.get(h) → 命中! 返回 block_id=7
  ✔ 跳过这一块，不需要 prefill

第 1 块: token_ids = [t4, t5, t6, t7]
  h = compute_hash([t4,t5,t6,t7], prefix=上一块的h)  # 链式！
  查表: hash_to_block_id.get(h) → 命中! 返回 block_id=12
  ✔ 跳过

第 2 块: token_ids = [t8, t9, t10, t11]
  h = compute_hash([t8,t9,t10,t11], prefix=上一块的h)
  查表: hash_to_block_id.get(h) → -1，没命中
  ✘ 停止！前 2 块可以复用，从第 2 块开始重新算 KV
```

对应到代码 [can_allocate()](../nanovllm/engine/block_manager.py)：

```python
h = -1
num_cached_blocks = 0
for i in range(seq.num_blocks - 1):      # 注意 -1：最后一块通常没填满，不查
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)  # 链式累计
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        break                            # 没命中（或哈希冲突保护）→ 停止
    num_cached_blocks += 1               # 命中，继续下一块
return num_cached_blocks
```

所以你的理解大体正确，只需修正一点：

| 你的理解 | 实际做法 |
|---|---|
| “从第一个 **token** 开始” | 从第一个 **块** 开始（块 = block_size 个 token） |
| “第一个 token 的哈希与第二个相加” | 前一块的哈希与**本块全部 token** 一起算 |
| “找到没命中的地方开始生产 KV” | ✔ 完全正确 |

**为什么粒度是块而不是单个 token？**

1. KV Cache 的分配单位就是块；你不可能「复用半块」
2. 块尺寸默认 256 token，哈希表项数 = 序列长 / 256，极小——查找很快
3. 只有填满的块才入表（半满的块还会继续被写入，哈希不稳定）

## 3.4 命中：can_allocate 与 allocate

新序列入场时，逐块尝试匹配：

```python
def can_allocate(self, seq):
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):                # 注意 -1：最后一块通常没填满，跳过
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                      # 哈希冲突保护：实际 token 也要相等
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                        # 已被占用，只 +1 引用，不消耗 free 块
    if len(self.free_block_ids) < num_new_blocks:
        return -1                                      # 块不够，等下一步
    return num_cached_blocks
```

#### Q: 代码里的变量都是什么意思？看不出在算什么

这段代码的**目的**是回答两个问题：
1. 这个新序列的前缀能复用多少块？（返回 `num_cached_blocks`）
2. 显存够不够分配剩下的块？（不够就返回 -1）

**变量通词表：**

| 变量 | 含义 | 类比 |
|---|---|---|
| `seq.num_blocks` | 这个序列总共需要多少块 | “这本书共多少页” |
| `h` | 当前链式哈希的累计值 | “我计算到哪了”的指针 |
| `num_cached_blocks` | 已经命中的前缀块数 | “能省多少块的 prefill” |
| `num_new_blocks` | 还需要从 free 队列拿多少新块 | “实际要花多少显存” |
| `block_id` | 查表命中的物理块编号 | “图书馆这本书放在第几号书架” |
| `seq.num_blocks - 1` | 循环范围去掉最后一块 | “最后一页没写完，先不管” |

**用一个具体例子走一遍：**

新序列总长 10 token，`block_size=4`，所以 `seq.num_blocks = 3`（块 0: 4 token，块 1: 4 token，块 2: 2 token）。

初始值：
```
h = -1
num_cached_blocks = 0    # 还没有块被复用
num_new_blocks = 3       # 最差情况需要 3 个新块
```

i=0（第 0 块）：
```
h = compute_hash([t0,t1,t2,t3], -1)
block_id = hash_to_block_id.get(h) → 假设命中，返回 7
✔ 命中！num_cached_blocks = 1
block_id=7 在 used_block_ids 中? 假设不在（在 free 里）
→ num_new_blocks 不变，仍然是 3
```

i=1（第 1 块）：
```
h = compute_hash([t4,t5,t6,t7], 上一步的h)
block_id = hash_to_block_id.get(h) → 假设命中，返回 12
✔ 命中！num_cached_blocks = 2
block_id=12 在 used_block_ids 中? 假设在（正被别人用着）
→ num_new_blocks -= 1 = 2   # 这个块只需 ref_count++，不占 free 名额
```

i 到 `num_blocks - 1 = 2` 就停止了（块 2 只有 2 个 token，没填满，不查）。

最终检查：
```
len(free_block_ids) >= num_new_blocks?  即 free 块数 >= 2?
如果是 → 返回 num_cached_blocks = 2（"能复用 2 块”）
如果不是 → 返回 -1（"显存不够，等下一步"）
```

**`num_new_blocks` 的精妙之处**

`num_new_blocks` 一开始 = 总块数（3）。每当某个块命中且**已在 used 里**（别人正在用），就 -1。为什么？

- 如果命中且在 used：只需 ref_count++，**不需要从 free 拿块**
- 如果命中但在 free：要从 free 拉回来，**还是占了 free 一个名额**（虽然不需要 prefill）
- 没命中：要从 free 拿一块并 reset，**也占 free 名额**

所以 `num_new_blocks` 代表的是：“这个序列需要从 free 队列里拿多少个块”。最后与 `len(free_block_ids)` 比，就能知道显存够不够。

`allocate` 时：
- 命中且已被占用的块：`ref_count++`
- 命中但在 free 里的块：从 free 拉到 used，`ref_count = 1`
- 没命中的块：走 `_allocate_block()` 分配新块

#### Q: "命中但在 free 里"与"没命中"有什么区别？

先理解一个块的三种状态：

```
物理块的生命周期：

[被序列使用中]  ──序列结束──▶  [在 free 队列，但内容还在]  ──被强制复用──▶  [被 reset 成空白块]
    ref_count > 0                    ref_count = 0                      新块，内容清空
    在 used_block_ids 中             在 free_block_ids 中                在 used_block_ids 中
    hash 有效                       hash 仍然有效！                   hash 被清
```

**"命中但在 free 里"是什么意思？**

就是中间那个状态。举个完整例子：

```
1. 序列 A 用了物理块 7，里面存着 [t0,t1,t2,t3] 的 KV
   blocks[7].ref_count = 1，在 used

2. 序列 A 完成 → deallocate → ref_count 降为 0 → 块 7 回到 free 队列
   但！blocks[7].hash 和 blocks[7].token_ids 没被清空！
   hash_to_block_id[那个 hash] 仍然指向 7

3. 新序列 B 入场，前缀与 A 相同，算出同样的 hash
   查 hash_to_block_id → 命中！返回 block_id=7
   但块 7 在 free 里（ref_count=0）→ "命中但在 free"
```

此时的处理：把块 7 从 free 拉回 used，ref_count 设为 1。**不需要重新计算 KV！** 因为里面的 K/V 数据还是对的——它从来没被擦除过。

**"没命中"又是什么意思？**

```
4. 另一个序列 C 入场，算出的 hash 在 hash_to_block_id 里查不到
   → 没命中，需要分配一个全新的块
   → 调用 _allocate_block():
       从 free 队列头部拿一个块（比如拿到了块 5）
       块 5 以前可能存着老数据，现在 block.reset() 清空它
       如果块 5 的旧 hash 还在表里，顺便删掉
   → 得到一个干净的空白块，后续 prefill 时再往里写 KV
```

**核心区别一张表**

| | 命中且在 used | 命中但在 free | 没命中 |
|---|---|---|---|
| KV 数据还在？ | ✔ | ✔ | ✘（会被 reset） |
| 需要重新 prefill？ | ✘ | ✘ | ✔ |
| 占用 free 名额？ | ✘ | ✘（拉回来就行） | ✔（消耗一个 free 块） |
| 处理方式 | ref_count++ | 从 free 拉回 used | _allocate_block() |

**一个类比**

想象图书馆：
- **命中且在 used**：这本书正在被人借着，你也要看 → 登记一下"又多一个人借了"
- **命中但在 free**：书已还回书架，但还没被拿走 → 直接从书架拿走，内容完整，不用重新印刷
- **没命中**：图书馆没这本书 → 拿一本空白本子重新印（prefill）

> 这也是为什么 3.5 节“释放时为什么不立即清空内容”的原因——不清空就能当"免费缓存"。只有当 free 队列被用尽、不得不拿这个块来装全新数据时，才会真正 reset。

[deallocate()](../nanovllm/engine/block_manager.py)：

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)   # 回 free 队列，但 hash 没清！
```

[_allocate_block()](../nanovllm/engine/block_manager.py)：

```python
block_id = self.free_block_ids.popleft()
block = self.blocks[block_id]
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]      # 真的要复用了，才把旧 hash 抹掉
block.reset()
```

→ **块在 free 队列里时，内容仍保留**。下次有相同前缀的请求来，`can_allocate` 还能命中并把它"复活"——这是免费的"缓存命中"。

只有 free 队列被耗尽、不得不真正复用某块时，才清掉旧哈希、重置内容。

## 3.6 一个时序例子

```
请求 1: "你是个有用的助手。问题：xxx"          ← prefill 5 块
   完成 → deallocate → 5 块进 free 队列(内容保留)

请求 2: "你是个有用的助手。问题：yyy"          ← 入场
   can_allocate:
     第 1 块 hash 命中 → 命中
     第 2 块 hash 命中 → 命中
     第 3 块 hash 命中 → 命中
     第 4 块 hash 命中 → 命中（"问题："仍在前缀里）
     第 5 块 hash 失配（"xxx" vs "yyy"）→ break
   num_cached_blocks = 4 → prefill 跳过这 4 块，直接从第 5 块开始算
```

## 3.7 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| 单一哈希字典，LRU = `deque` 顺序 | 同思路，有更精细的 evict 策略 |
| 哈希函数：xxh64 | xxh64，相同 |
| 命中后比对 `token_ids` 防冲突 | 同样有冲突检查 |
| 不区分 LoRA | 哈希 key 会带 `lora_id` |
| 仅自动 caching | 还支持显式的"prompt cache" API |

## 3.8 动手任务

1. 用同一个 prompt 连续 `generate` 两次，在 [can_allocate](../nanovllm/engine/block_manager.py) 里加打印，观察第二次的 `num_cached_blocks`
2. 故意在两个 prompt 前面拼相同 1000 token，对比 prefill 时间
3. 思考：如果两条序列共享前缀但温度不同，共享 KV 是否安全？
   *答：安全。KV 只与 token 内容相关，与采样温度无关。*
4. 思考：如果存在哈希冲突（不同 token 序列产生相同 xxh64），会发生什么？
   *答：`can_allocate` 中的 `self.blocks[block_id].token_ids != token_ids` 兜底，不会数据污染，最多是 cache miss。*

---

上一章 ← [02 · PagedAttention](./02-paged-attention.md) | 下一章 → [04 · 调度器](./04-scheduler.md)
