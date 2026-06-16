# 08 · 采样器与权重加载

## 学习目标

- Gumbel-Max 是怎么做到"无分支"温度采样的？
- HF 权重的 `q_proj/k_proj/v_proj` 如何被合并成 `qkv_proj`？
- 为什么每个参数都自带一个 `weight_loader`？

## 涉及源码

- [nanovllm/layers/sampler.py](../nanovllm/layers/sampler.py)
- [nanovllm/utils/loader.py](../nanovllm/utils/loader.py)
- [nanovllm/models/qwen3.py](../nanovllm/models/qwen3.py)（`packed_modules_mapping`）
- [nanovllm/layers/linear.py](../nanovllm/layers/linear.py)（各种 `weight_loader`）

## 8.1 Sampler：一行 Gumbel-Max

[Sampler.forward](../nanovllm/layers/sampler.py)：

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs  = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

### 8.1.1 数学背景

要从分布 `p` 采样一个 token：

经典做法：累计分布 + 二分查找——有分支，对 GPU 不友好。

**Gumbel-Max trick**：
- 给每个 logit 加上独立的 Gumbel(0,1) 噪声
- 取 argmax → 等价于按 softmax 概率采样

等价变形（这里用 Exp(1) 形式）：
```
sample = argmax(probs / Exp(1) noise)
       = argmax(softmax(logits) / Exp(1))
```

### 8.1.2 为什么这样写性能极好

- **完全张量化**，无 Python 分支
- 所有运算 in-place（`div_`、`clamp_min_`），减少显存分配
- 整段塞进 `@torch.compile`，融合成一个 kernel

> 真实 vLLM 的 Sampler 复杂得多（top-k / top-p / min-p / 频率惩罚 / logit bias / logprobs / Beam Search），但骨架就是这样。

### 8.1.3 一个细节：为什么不允许 `temperature=0`

[sampling_params.py](../nanovllm/sampling_params.py)：

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

因为 `logits / 0` 会出现 nan。真要贪心采样，把温度设很小（如 1e-6）即可，效果等同 argmax。

## 8.2 权重加载：`weight_loader` 模式

### 8.2.1 问题：HF checkpoint vs 推理时的合并参数

HuggingFace 的 Qwen3 checkpoint 中有：
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.self_attn.k_proj.weight`
- `model.layers.0.self_attn.v_proj.weight`
- `model.layers.0.mlp.gate_proj.weight`
- `model.layers.0.mlp.up_proj.weight`

但 nano-vllm 中模型只有：
- `model.layers.0.self_attn.qkv_proj.weight`（合并版）
- `model.layers.0.mlp.gate_up_proj.weight`（合并版）

需要在加载时把三个 / 两个权重"塞"进合并后的参数。

### 8.2.2 解法 1：每个参数自带 `weight_loader`

[LinearBase.__init__](../nanovllm/layers/linear.py)：

```python
self.weight = nn.Parameter(torch.empty(output_size, input_size))
self.weight.weight_loader = self.weight_loader   # 函数附在 Parameter 上
```

不同子类的 `weight_loader` 各自实现 TP 切片 + 偏移逻辑，例如 [QKVParallelLinear.weight_loader](../nanovllm/layers/linear.py)：

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    if loaded_shard_id == "q":
        shard_size, shard_offset = num_heads * head_size, 0
    elif loaded_shard_id == "k":
        shard_size = num_kv_heads * head_size
        shard_offset = num_heads * head_size
    else:  # "v"
        shard_size = num_kv_heads * head_size
        shard_offset = num_heads * head_size + num_kv_heads * head_size
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    param_data.copy_(loaded_weight)
```

`narrow` 圈出合并参数中"q/k/v 应该住的那一段"，`chunk` 取出本卡的 TP 切片，`copy_` 写入。

### 8.2.3 解法 2：模型类提供 `packed_modules_mapping`

[Qwen3ForCausalLM.packed_modules_mapping](../nanovllm/models/qwen3.py)：

```python
packed_modules_mapping = {
    "q_proj":     ("qkv_proj", "q"),
    "k_proj":     ("qkv_proj", "k"),
    "v_proj":     ("qkv_proj", "v"),
    "gate_proj":  ("gate_up_proj", 0),
    "up_proj":    ("gate_up_proj", 1),
}
```

这张表说："看到 weight 名里带 `q_proj`，就把它当作 `qkv_proj` 的 `q` 分片处理"。

### 8.2.4 加载流程

[loader.py](../nanovllm/utils/loader.py)：

```python
def load_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

逻辑：
1. 遍历每个 safetensors 文件的每个 weight
2. 名字命中 `packed_modules_mapping` → 改名 + 按 `shard_id` 调用 `weight_loader`
3. 否则走默认 `weight_loader`（`ColumnParallelLinear` 自动 TP 切片，`RowParallelLinear` 同理）

注意 `for...else` 语法：`for` 没被 `break` 时执行 `else`。

## 8.3 关键设计：把切片逻辑下沉到参数

为什么不集中在 `load_model` 里写一个大 if-else？

- 模型种类多：Qwen / Llama / Mistral / DeepSeek / …
- 每种模型的合并方式可能不同
- 集中式逻辑会爆炸

vLLM 的做法是：**让每个 `nn.Parameter` 自己知道怎么接收权重**。`load_model` 只负责调度，具体逻辑由各 `Linear` 子类的 `weight_loader` 决定。这是非常标准的 OO 模式。

## 8.4 与真实 vLLM 的对比

| nano-vllm | 真实 vLLM |
|---|---|
| Sampler 仅温度采样 | top-k/top-p/min-p/频率惩罚/logit bias 等 |
| 仅 safetensors | 还支持 .bin / .pt / NPU 格式 |
| 加载 CPU → cuda 默认设备 | 支持权重磁盘 mmap、量化加载、tensor sharding |
| 单一 `packed_modules_mapping` | 同思路，每个模型类提供自己的 |
| 不支持 LoRA | 支持，并需要相应的 `lora_loader` |

## 8.5 动手任务

1. 把温度从 0.6 改到 1e-6，对比生成结果（应该接近 argmax）
2. 在 [load_model](../nanovllm/utils/loader.py) 里 print 每个权重的形状与目标参数的形状，自行验证 TP 切片逻辑
3. 思考：如果 HF checkpoint 里 `q_proj/k_proj/v_proj` 形状不一致（GQA），合并后的 `qkv_proj.weight` 行数应是多少？
   *答：`(num_heads + 2*num_kv_heads) * head_size`，和 [QKVParallelLinear.__init__](../nanovllm/layers/linear.py) 里写的一致。*
4. 自己尝试给 Sampler 加一个 top-k：
   *提示：`logits.scatter_(-1, low_k_indices, -inf)` 即可，注意这破坏了 `@torch.compile` 的形状静态性。*

---

上一章 ← [07 · CUDA Graph](./07-cuda-graph.md) | 下一章 → [09 · 与真实 vLLM 的差异和进阶路线](./09-vs-real-vllm.md)
