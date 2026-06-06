# GRPO Training Pipeline — 数据流与架构审计

> 审计日期: 2026-06-04
> 审计范围: `medrl/rl/grpo_trainer.py`, `medrl/rl/advantage.py`, `medrl/rl/kl_controller.py`, `medrl/rl/reward_func.py`, `scripts/train_phase2.py`

---

## 一、完整数据流向图（伪代码）

```python
# ============================================================
# 输入: B=2 个 prompt 字符串
# ============================================================
prompts = [
    "A 45-year-old male presents with...\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\n\n"
    "Correct answer: C. Lisinopril\n\nExplain step-by-step...",

    "A 32-year-old female with history of...\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\n\n"
    "Correct answer: A. Prednisone\n\nExplain step-by-step...",
]

# ============================================================
# STEP 1: 组采样 (Group Sampling) — 1 个 prompt → G 个响应
# ============================================================
# _generate_responses() 内部逻辑:
format_prefix = (
    "You MUST respond in this exact format:\n"
    "<thinking>\nStep 1: ...\n</thinking>\n"
    "<answer>\n...\n</answer>"
)
# 每个 prompt 复制 G=4 次, 得 B*G=8 个带格式前缀的完整prompt
all_prompts = [format_prefix + p for p in prompts for _ in range(4)]

# 左padding → tokenize → policy.generate(do_sample=True, temperature=0.8)
# → 8 次独立采样 (同一 prompt 的 4 份因 temperature/top_p 随机性产生不同响应)
# → decode 仅保留新生成的 token → 8 个 response 字符串
responses = [
    "<thinking>\nStep 1: ...\n</thinking>\n<answer>C</answer>",  # prompt_0, sample 1
    "<thinking>\nStep 1: ...\n</thinking>\n<answer>B</answer>",  # prompt_0, sample 2
    "<thinking>\nStep 1: ...\n</thinking>\n<answer>C</answer>",  # prompt_0, sample 3
    "<thinking>\nStep 1: ...\n</thinking>\n<answer>A</answer>",  # prompt_0, sample 4
    # ... prompt_1 的 4 个响应
]

# ============================================================
# STEP 2: 计算 Old Log Probs（冻结，不参与梯度）
# ============================================================
# _compute_log_probs() — 【审计前有 Bug, 见下文第四节】
#   prompt + response 拼接 → tokenize → policy.forward() → logits
#   logits 形状: (B*G=8, seq_len, vocab_size=152064)
#
#   【错误实现】logits.log_softmax(dim=-1).mean(dim=[1,2])
#     → 对所有 token 位置、所有词表条目的 log-prob 取均值 → 一个标量
#     → 物理含义: 模型在所有位置的平均预测熵, 不是生成这段文本的概率!
#
#   【正确实现】对每个 response token 位置, gather 其对应 token ID 的 log-prob,
#     然后 sum over 序列 → 得到每段 response 的序列级 log 概率
with torch.no_grad():
    log_probs_old = compute_log_probs(expanded_prompts, responses).detach()
# 形状: (8,) — 8个标量 (每个 response 一个序列级 log-prob)

# ============================================================
# STEP 3: 计算 Reward（结果监督 — 非过程监督!）
# ============================================================
# reward_func() 对每个 (prompt, response) 对计算复合奖励:
#
#   Component 1: format_reward(response)
#     - 检查 <thinking> </thinking> <answer> </answer> 四个标签是否存在
#     - 每个标签 0.25 分, 最高 1.0
#     - 空 <answer> 标签额外扣 0.5
#
#   Component 2: judge_reward(question, response)
#     - 调用 DeepSeek API (或任何 OpenAI-compatible 端点)
#     - 对诊断正确性/逻辑连贯性/术语准确性 三个维度打分 0~1
#     - 无 API 时回退到 _heuristic_score()
#
#   Component 3: diversity_reward(response)
#     - n-gram Type-Token Ratio (n=3)
#     - 检测重复模式 (20+ char 重复3次以上 → 直接给0)
#
#   Component 4: _check_correctness(response, gold_answer)
#     - 从 <answer> 标签提取预测, 与 ground truth 匹配
#     - 精确选项字母匹配 → 1.0, 部分文本匹配 → 0.8/0.5, 不匹配 → 0.0
#
#   最终 reward = weighted_sum / total_weight
#     w_format=0.2 + w_judge=0.65 + w_diversity=0.15 + w_correct=0.3 = 1.3
rewards = [0.72, 0.15, 0.81, 0.43,   # prompt 0 的 4 个响应
           0.55, 0.91, 0.38, 0.67]   # prompt 1 的 4 个响应
# 形状: (8,)  — 每个 response 一个标量 reward

# ============================================================
# STEP 4: 计算 Group-Relative Advantage（组内相对优势）
# ============================================================
# compute_grpo_advantage():
#   rewards_2d = rewards.view(2, 4)
#   = [[0.72, 0.15, 0.81, 0.43],   ← prompt_0 组
#      [0.55, 0.91, 0.38, 0.67]]   ← prompt_1 组
#
#   mu  = mean per row:  [0.5275, 0.6275]
#   std = std per row:   [0.262,  0.191]
#
#   advantages_2d = (rewards_2d - mu) / std
#   = [[+0.73, -1.44, +1.08, -0.37],   ← prompt_0 组内相对排名
#      [-0.41, +1.48, -1.30, +0.22]]   ← prompt_1 组内相对排名
#
#   展平回 (8,)
#
#   ⚠️ 关键性质:
#   - 每个 advantage 是一个标量, 对应一个完整的 response
#   - 同一组内 advantages 和为 0 (因为减去了组均值)
#   - 无需 Critic 网络 — 优势信号完全来自组内相对比较

# ============================================================
# STEP 5: PPO 更新（默认 1 个 epoch）
# ============================================================
# 重新计算 current log_probs (带梯度)
log_probs = compute_log_probs(expanded_prompts, responses)  # (8,) 标量, 有 grad

ratio = exp(log_probs - log_probs_old)  # (8,) 标量 ratio per response
# ⚠️ 标量 ratio = exp(Σ log_p_θ(token) - Σ log_p_old(token))
#               = Π p_θ(token) / Π p_old(token)
# 这是整个 response 序列的重要性采样比, 不是逐 token 的

surr1 = ratio * advantages               # 非裁剪目标
surr2 = clamp(ratio, 0.8, 1.2) * advantages  # PPO 裁剪目标
policy_loss = -min(surr1, surr2).mean()  # 1 个标量 loss

policy_loss.backward()  # 梯度通过 log_probs → logits → LoRA 参数
optimizer.step()        # 仅更新 LoRA adapter 参数

# ============================================================
# STEP 6: KL 监控 → 【硬编码为 0!】
# ============================================================
kl_div = 0.0     # ⚠️ 实际未计算, 注释说 "48GB VRAM too tight for 7B × 2 forward"
entropy = 0.0    # ⚠️ 实际未计算
beta = kl_controller.beta  # ⚠️ beta 从未被 step() 更新, 始终是初始值 0.01
```

---

## 二、Q1: Prompt 如何变成 G 个样本？

```
prompt_0  ─┬→ [复制1 + 格式前缀] → prompt_0'  ─→ sample(t=0.8) → response_0a
           ├→ [复制2 + 格式前缀] → prompt_0'  ─→ sample(t=0.8) → response_0b
           ├→ [复制3 + 格式前缀] → prompt_0'  ─→ sample(t=0.8) → response_0c
           └→ [复制4 + 格式前缀] → prompt_0'  ─→ sample(t=0.8) → response_0d

prompt_1  ─┬→ [复制1 + 格式前缀] → prompt_1'  ─→ sample(t=0.8) → response_1a
           ├→ [复制2 + 格式前缀] → prompt_1'  ─→ sample(t=0.8) → response_1b
           ├→ [复制3 + 格式前缀] → prompt_1'  ─→ sample(t=0.8) → response_1c
           └→ [复制4 + 格式前缀] → prompt_1'  ─→ sample(t=0.8) → response_1d
```

**核心机制**：
1. 每个原始 prompt 前注入格式指令 (`You MUST respond in this exact format: <thinking>...</thinking><answer>...</answer>`)
2. 每个处理后的 prompt 复制 G=4 次
3. B*G 个 prompt 打包成一个 batch，一次性调用 `policy.generate(do_sample=True, temperature=0.8)`
4. 因为 `do_sample=True`，随机采样核导致同一 prompt 的 G 份产生 G 条不同的 response
5. `tokenizer.padding_side = "left"` 确保自回归生成的 attention 对齐正确
6. 解码时仅保留 `output_ids[:, input_len:]`（新生成的 token），丢弃 prompt 部分

**关键参数**：
- `temperature=0.8`（Phase 2 实际用 1.0）：越高多样性越大，组内方差越大
- `top_p=0.9`：nucleus sampling 截断，滤除低概率尾部分布
- `max_new_tokens=1024`：单次生成最长 1024 token

---

## 三、Q2 & Q3: 优势函数维度 + 参数冻结状态

### Q2: A_i 是 Token 级还是 Sequence 级？

**答案：Sequence 级。每个 response 只有一个标量 advantage。**

证据链：
```python
# reward_func()      → rewards:    (B*G,)   每个response一个标量reward
# advantage.py       → advantages: (B*G,)   每个response一个标量advantage
# grpo_trainer.py L245-254:
ratio = (log_probs - log_probs_old).exp()  # (B*G,) 标量ratio
surr1 = ratio * advantages                  # 标量 × 标量
surr2 = clamp(ratio, 0.8, 1.2) * advantages
policy_loss = -min(surr1, surr2).mean()
```

**这意味着什么**：梯度下降时，好 response 的**所有 token** 被统一放大生成概率，差 response 的**所有 token** 被统一压低。模型无法区分 "前几步推理正确但最后答案错了" 和 "从头到尾都在胡说" — 缺乏 per-token 的细粒度信用分配（credit assignment）。

### Q3: 哪些参数冻结？哪些在动？

```
┌──────────────────────────────────────────────────────────┐
│              Policy Model (Qwen2.5-7B + LoRA)             │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  基座权重                                            │  │
│  │  W_q, W_k, W_v, W_o, gate, up, down, embed,        │  │
│  │  lm_head, layernorm, ...                           │  │
│  │  dtype: float16                                     │  │
│  │  requires_grad = False  ← ❄️ 完全冻结               │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  LoRA Adapter (r=16, alpha=32)                      │  │
│  │  q_proj:  A_q ∈ R^{d×16}, B_q ∈ R^{16×d}           │  │
│  │  k_proj:  A_k ∈ R^{d×16}, B_k ∈ R^{16×d}           │  │
│  │  v_proj:  A_v ∈ R^{d×16}, B_v ∈ R^{16×d}           │  │
│  │  o_proj:  A_o ∈ R^{d×16}, B_o ∈ R^{16×d}           │  │
│  │  gate_proj: A_g, B_g                               │  │
│  │  up_proj:   A_u, B_u                               │  │
│  │  down_proj: A_d, B_d                               │  │
│  │  requires_grad = True  ← 🔥 唯一在训练的参数          │  │
│  │  总可训练参数量: ~40M (占 7B 的 ~0.6%)               │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│          Reference Model (Qwen2.5-7B, 无 LoRA)            │
│  全部基座权重: model.eval(), requires_grad=False           │
│  dtype: float16                                           │
│  ❄️ 完全冻结, 仅用于 KL 散度计算 (但当前未实际参与训练)     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  Optimizer (AdamW)                        │
│  optimizer = AdamW([p for p in model.parameters()         │
│                      if p.requires_grad])                 │
│  仅持有 LoRA 参数引用, 基座权重不在 optimizer 中            │
└──────────────────────────────────────────────────────────┘
```

**显存分布 (FP16, 单卡 48GB)**：

| 组件 | 大小 |
|---|---|
| Policy 基座权重 (FP16) | ~14 GB |
| LoRA Adapter | ~0.08 GB |
| Reference 模型 (FP16) | ~14 GB |
| Optimizer States (LoRA only) | ~0.3 GB |
| Activations (grad ckpt) | ~8 GB |
| Group Responses (G=4) | ~6 GB |
| **总计** | **~44 GB** |

---

## 四、审计发现的三个关键问题

### 问题 1 (严重): `_compute_log_probs` 计算的是熵, 不是生成概率

**位置**: `grpo_trainer.py:193-194`

```python
# 错误实现:
log_probs = outputs.logits.log_softmax(dim=-1)
return log_probs.mean(dim=[1, 2])  # 对所有token和所有词表条目取均值!
```

这段代码取 `log_softmax` 后在 `dim=[1,2]` 上取均值。`dim=1` 是序列长度, `dim=2` 是词表大小 (152064)。所以它计算的是 **"模型在所有位置、对所有词汇的平均 log 概率"** — 这本质上是负的平均预测熵, 不是模型实际生成这些 token 的 log 概率。

**正确做法**: 对每个 response token 位置, gather 该位置实际 token ID 对应的 log-prob, 然后 sum over 序列:

```python
# 正确实现:
# 1. logits → log_softmax
# 2. 找到 response token 的起始位置
# 3. gather 实际 token ID 的 log-prob
# 4. sum over 序列维度 → 得到序列级 log P(response | prompt)
```

**影响**: `ratio = exp(log_probs - log_probs_old)` 不反映策略更新前后对同一段 response 的真实概率变化。PPO clipping 在优化一个无意义的目标。

### 问题 2 (中等): KL 散度未参与训练

**位置**: `grpo_trainer.py:270-272`

```python
kl_div = 0.0     # 硬编码为0, 注释说显存不够
entropy = 0.0
beta = self.kl_controller.beta  # 始终是初始值, 从未更新
```

`KLController.step()` 从未被调用, `kl_beta` 从未被使用。policy loss 中完全没有 KL 正则项。这意味着：
- 没有任何机制阻止 policy 偏离 reference model 太远
- 模型可能发生 reward hacking 式退化（重复输出、格式崩溃等）
- `beta` 始终停留在初始值 0.01, 形同虚设

### 问题 3 (设计): Sequence-level advantage 无法支撑 PRM（过程监督）

当前整个 pipeline 的 reward 和 advantage 都是 response-level 的。要实现 CLAUDE.md 中描述的 "过程监督 (PRM)", 需要 per-step 或 per-token 的 reward 信号。这需要在数据层面构造分步标注, 并在训练层面支持 per-step advantage 计算。

---

## 五、修复方案

### Fix 1: 重写 `_compute_log_probs` — 正确计算序列 log 概率

- Tokenize prompt+response 拼接文本
- 单独 tokenize prompt 获取每个样本的 prompt 长度
- Forward pass 获取 logits
- Shift logits (logits[t] 预测 token[t+1])
- Gather 每个 response token 位置的实际 log-prob
- Sum over 序列 → 得到正确的序列级 log P(response | prompt, θ)

### Fix 2: 加入 KL 散度计算与惩罚

- 对 reference model 做一次 forward pass (no_grad) 计算 log P_ref(response | prompt)
- 在 PPO loss 中加入 `beta * KL(π_θ || π_ref)` 项
- 每步调用 `KLController.step()` 自适应调节 beta
- 日志输出真实的 KL 值

### Fix 3: 验证梯度流

- 确保 ratio 从 sum-of-logprobs 正确计算
- 确保 backward 时梯度能通过 gather 操作回传到 logits → LoRA 参数
