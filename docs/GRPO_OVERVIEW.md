# GRPO 算法数学原理与架构深度解析

## 1. 为什么 GRPO 可以彻底抛弃 Critic 网络？

### 1.1 PPO 的显存痛点

标准 PPO（Proximal Policy Optimization）需要同时维护 4 个模型：

| 模型 | 作用 | 显存占比（7B 为例） |
|------|------|---------------------|
| Policy Model (Actor) | 生成响应 | ~14 GB (FP16) |
| Reference Model | KL 散度约束锚点 | ~14 GB |
| **Critic Model** | 估计状态价值 V(s) | **~14 GB** |
| Reward Model | 提供标量奖励信号 | ~14 GB |

PPO 计算优势函数的经典公式（GAE）：

```
A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

**V(s) 必须由 Critic 网络预测**，这就是 PPO 显存爆炸的根源。单张 RTX 4090（24 GB）连一个 7B 模型都装不下，更别说训练。

### 1.2 GRPO 的核心洞见：组内相对比较替代绝对价值估计

GRPO 的关键创新：**对于同一个 prompt q，采样 G 个不同响应 {o_1, ..., o_G}，用这 G 个响应的奖励分布来直接计算优势，完全绕过 Critic。**

直觉理解：

```
PPO 思路：   "这条轨迹有多好？" → 需要 Critic 估算绝对价值
GRPO 思路：  "这条轨迹比组内平均好多少？" → 只需要组内相对排名
```

类比：给一个班级的学生排名，不需要知道每个学生的"绝对 IQ 值"，只需要知道他们这次考试分数的相对位置。**组均值 μ 就是那个"班级平均分"。**

---

## 2. 数学公式与代码级计算流程

### 2.1 核心公式

对于 prompt q，采样 G 个响应 {o_i}，获取 G 个奖励 {R_i}：

```
μ_group = (1/G) · Σ R_i          — 组内均值
σ_group = √[(1/G) · Σ (R_i - μ)²]  — 组内标准差
A_i = (R_i - μ_group) / σ_group   — 组内标准化优势
```

GRPO 的策略梯度目标函数：

```
J_GRPO(θ) = E_{q ~ P(Q), {o_i} ~ π_old}[
    1/G · Σ_i min(
        r_i(θ) · A_i,
        clip(r_i(θ), 1-ε, 1+ε) · A_i
    )
    - β · D_KL(π_θ || π_ref)
]
```

其中 r_i(θ) = π_θ(o_i|q) / π_old(o_i|q) 是概率比，与 PPO 一致。

### 2.2 逐行代码级计算流程

```python
def grpo_training_step(policy, ref_model, reward_func, prompts, group_size=8):
    """
    单步 GRPO 训练 — 完全不依赖 Critic 网络。
    
    Args:
        policy:     当前策略模型
        ref_model:  冻结的参考模型（用于 KL）
        reward_func:复合奖励函数
        prompts:    一批 prompt
        group_size: 每个 prompt 采样多少条响应（G）
    """
    B = len(prompts)

    # ── Step 1: 组内采样 ——
    # 每个 prompt 独立采样 G 条响应
    # 形状: (B * G, max_seq_len)
    all_prompts = prompts.repeat_interleave(group_size, dim=0)
    responses = policy.generate(all_prompts)  # 自回归采样

    # ── Step 2: 计算奖励（无需 Critic！）——
    # 形状: (B * G,)
    rewards = reward_func(all_prompts, responses)

    # ── Step 3: 组内标准化得到优势 ——
    # 重塑为 (B, G)，按组计算 μ 和 σ
    rewards_grouped = rewards.view(B, group_size)        # (B, G)

    mu = rewards_grouped.mean(dim=-1, keepdim=True)      # (B, 1)
    std = rewards_grouped.std(dim=-1, keepdim=True)      # (B, 1)
    std = std.clamp(min=1e-4)  # 数值稳定：防止除零

    advantages = (rewards_grouped - mu) / std            # (B, G)
    advantages = advantages.view(-1)                     # (B*G,)

    # ── Step 4: 计算 log_prob 比率 ——
    log_probs_old = policy.compute_log_prob(all_prompts, responses).detach()
    log_probs_new = policy.compute_log_prob(all_prompts, responses)
    ratio = (log_probs_new - log_probs_old).exp()        # r_i(θ)

    # ── Step 5: PPO-style Clipped Loss ——
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── Step 6: KL 散度正则（防策略崩塌）——
    # 直接用 ref_model 输出 logits 做 closed-form KL
    with torch.no_grad():
        ref_logits = ref_model(all_prompts).logits
    policy_logits = policy(all_prompts).logits

    # KL(π_θ || π_ref) = Σ π_θ(y) · [log π_θ(y) - log π_ref(y)]
    kl_div = F.kl_div(
        F.log_softmax(policy_logits, dim=-1),
        F.softmax(ref_logits, dim=-1),
        reduction='batchmean'
    )

    # ── Step 7: 总损失 ——
    total_loss = policy_loss + beta * kl_div

    return total_loss
```

### 2.3 关键观察：Critic 在哪里？

**没有。** 整个计算流程只涉及：

1. `policy` — 生成 + 计算 log_prob
2. `ref_model` — 冻结，仅用于 KL 计算
3. `reward_func` — 标量奖励函数（规则 + LLM Judge）

对比 PPO 需要的 `critic.forward(state) → V(s)`，GRPO 的优势完全来自**同一 prompt 下不同响应之间的奖励相对差**。

---

## 3. 组大小 G 的边界效应深度分析

### 3.1 数值稳定性

当 G 较小时，σ_group 的估计极不稳定：

```
G=2:  σ 只有两个数据点，二值化严重
      → 优势退化为 sign(R_1 - R_2) / 常数
      → 损失大量信息，训练信号粗糙

G=4:  勉强可用，但 σ 的变异系数约 ~50%
      → 部分 batch 的优势估计偏差较大

G=8:  推荐起点，σ 相对稳定
      → 每 prompt 8 条响应的显存开销可控

G=16: 统计最优，但显存 ×2
      → 适合 A100 80G 等大显存场景
```

### 3.2 奖励同质化的问题

如果复合 Reward 设计不当，导致同 prompt 下 G 个响应得分几乎相同：

```
R = [0.72, 0.71, 0.73, 0.72, 0.71, 0.72, 0.73, 0.71]
σ ≈ 0.007  →  A = (R - 0.719) / 0.007
```

此时 **优势退化为对随机噪声的放大**，策略梯度方向失去意义。这是工程上最致命的失败模式。

**解决方案**（我们后续在 `reward_func.py` 中实现）：
- **硬约束拉开差距**：格式违规直接 -1.0，制造天然的高方差
- **LLM Judge 连续打分**：使用 0-1 连续分而非离散 0/1
- **多样性奖励**：对生成内容的 n-gram 多样性给予微量正分

---

## 4. KL 散度与熵正则的动态调节机制

### 4.1 KL 散度：防止策略跑偏

训练中持续监控：

```
KL(π_θ || π_ref) = E_y~π_θ [log π_θ(y|x) - log π_ref(y|x)]
```

当 KL 超过阈值（通常 0.02-0.05），增大 β 系数或暂停更新。

### 4.2 熵正则：防止生成崩塌

策略熵衡量输出的多样性：

```
H(π_θ) = -Σ_y π_θ(y|x) · log π_θ(y|x)
```

低熵 → 模型只会生成固定模式（如反复输出"让我再想想"）→ Reward Hacking 的征兆。

将熵奖励加入总 Reward：
```
R_final = R_task + λ_entropy · H(π_θ)
```

### 4.3 KL Controller 设计（自适应 β）

```python
class KLController:
    def __init__(self, target_kl=0.02, beta=0.01, lr=0.1):
        self.target_kl = target_kl
        self.beta = beta
        self.lr = lr

    def step(self, current_kl):
        """PID-style KL 自适应调节。"""
        error = current_kl - self.target_kl
        # 乘法更新：KL 过高则增大惩罚，过低则放松
        self.beta *= (1 + self.lr * error)
        self.beta = max(1e-4, min(1.0, self.beta))
        return self.beta
```

---

## 5. PPO vs GRPO 架构对比

```
PPO (Actor-Critic):
┌──────────┐    ┌──────────┐
│  Actor   │    │  Critic  │
│ π_θ(a|s) │    │ V_φ(s)   │
└────┬─────┘    └────┬─────┘
     │               │
     │  sample       │  estimate V(s)
     ▼               ▼
┌──────────┐    ┌──────────┐
│  GAE:    │ =  │  r + γV  │  ← Critic 必须参与
│  A_t     │    │  - V(s)  │
└──────────┘    └──────────┘

GRPO (Group-Relative):
┌──────────┐
│  Actor   │
│ π_θ(a|s) │  ← 唯一的训练参数！
└────┬─────┘
     │
     │ sample G responses per prompt
     ▼
┌──────────────────────────────┐
│  A_i = (R_i - μ_group) / σ   │  ← 纯组内统计，零额外参数
│  μ, σ 来自同一 prompt 的     │
│  G 个响应的奖励分布           │
└──────────────────────────────┘
```

**显存对比（7B 模型, FP16）：**

| 组件 | PPO | GRPO | 节省 |
|------|-----|------|------|
| Policy (Actor) | 14 GB | 14 GB | — |
| Reference | 14 GB | 14 GB | — |
| **Critic** | **14 GB** | **0** | **-14 GB** |
| Reward Model | 14 GB | 14 GB | — |
| **合计** | **56 GB** | **42 GB** | **-25%** |
| 可选：RM→LLM Judge (API) | — | 可移出 GPU | 再 -14 GB |

---

## 6. DAPO：GRPO 的进一步增强

DAPO（Dual-Agent Policy Optimization）在 GRPO 基础上引入：
- **双策略交互**：主策略 + 辅助策略同时采样，相互提供对比信号
- **动态组大小**：根据奖励方差自适应调整 G
- **Clip 策略增强**：对极端优势值进行二次裁剪，防梯度爆炸

我们后续在实现 GRPO 后，可逐步引入 DAPO 改进。

---

## 7. 总结：GRPO 的工程优势

1. **省显存**：不需要 Critic，7B 模型可在单张 A6000（48G）甚至 RTX 4090（24G，配合 ZeRO-3 offload）上训练
2. **训练稳定**：组内标准化自带归一化效果，不需精心调 GAE 的 γ/λ
3. **天然抗 Reward Hacking**：组内相对比较意味着"大家都在骗长度奖励时，骗得少的反而吃亏更小"
4. **实现简洁**：核心逻辑 ~100 行 PyTorch，比 PPO+GAE 更容易调试
