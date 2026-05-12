# Hierarchical Reasoning Library: Co-Evolving Abstraction Memory for LLM Reasoning

## 1. Problem Setting

构建一个 **hierarchical reasoning library**，包含从解题经验中提取的抽象提醒条目，每条标记为 strategy（正面策略）或 caution（警示提醒）。通过 **逐题提取 → 跨题聚合** 的两步流程构建 library，按 general → domain → problem-specific 三层组织。在 RL 训练过程中，library 和模型 **co-evolve**：模型变强后犯错模式改变，library 随之更新。

```
Cold Start:
  base model 做题 → 大模型逐题分析 trace → 跨题聚合 → library v1
  用 library v1 辅助的正确解题 trace 做 SFT → model v0（学会如何使用 library）

Co-Evolution Loop:
  Round 1: model v0 + library v1 辅助 → GRPO 训练 → model v1
  Round 2: model v1 做题 → 大模型逐题分析 → 跨题聚合 → library v2
           model v1 + library v2 辅助 → GRPO 训练 → model v2
  Round 3: ...（validation-triggered：性能停滞时才更新 library）
```

### 核心科学问题

1. **Co-evolution 是否优于 static library？** 随模型进步动态更新 library 是否比固定 library 更好？
2. **Hierarchical 结构是否优于 flat retrieval？** general → domain → problem-specific 的分层检索是否比简单 top-K 更有效？
3. **Caution 条目的价值**：从失败经验中抽象出的警示提醒是否是 library 中不可或缺的成分？
4. **简单任务产出的 abstraction 在 library 中扮演什么角色？** 是否自然形成 general 层的主要贡献？

---

## 2. 与已有工作的关系

### 最相关文献

| 论文 | 时间 | 做了什么 | 与本工作的差异 |
|---|---|---|---|
| **RLAD** (Qu et al.) | 2025.10 | RL 训练中联合优化 abstraction generator + solution generator | 即时生成 abstraction，无 library 无检索；单 abstraction；无 co-evolution |
| **Metacognitive Reuse** (Didolkar et al.) | 2025.09 | 从推理 trace 提取 behavior → handbook → embedding 检索 | Flat library 无层级；只含成功策略无 caution；后处理不在训练循环中 |
| **Dynamic Cheatsheet** (Suzgun et al.) | 2025.04 | Test-time 边做边积累 evolving memory | 不涉及 RL 训练；不区分层级；memory 从 test 本身来 |
| **ExpeL** (Zhao et al.) | 2023.08 | Agent 从 training tasks 提取 insight + 检索 | Agent 任务非 reasoning；flat insight list；不在 RL 循环中 |
| **SkillRL** (Xia et al.) | 2026.02 | Hierarchical SkillBank + recursive evolution during RL | Agent 场景非数学推理；最接近的 co-evolution 设计 |
| **LeMa** (An et al.) | 2024 | 用 LLM 的 "错误+纠正" 做 SFT | SFT 不是 retrieval；包含具体解题步骤有 leakage 风险 |
| **Can LLMs Learn from Mistakes?** (Tong et al.) | 2024 ACL | Mistake tuning + self-rethinking | 训练时直接用错误 trace，不抽象为通用提醒 |
| **SKiC** (Chen et al.) | 2024 ACL | Prompt 中提供 skill + 组合 demo | 纯 prompting；skill 人工提供无自动提取 |
| **E2H Reasoner** | 2025.06 | Curriculum RL 从简单到困难 | 无 abstraction 机制；隐式 easy-to-hard |

### Novelty 分析

**已有人做的（单独看不新）：**
- 从 reasoning trace 提取 abstraction → RLAD、Metacognitive Reuse
- 外部 library + embedding 检索辅助推理 → Dynamic Cheatsheet、ExpeL
- 从错误中学习 → LeMa、Tong et al.
- Hierarchical skill library → SkillRL、H-MEM
- Library 与 agent co-evolve → SkillRL（agent 场景）

**没有人做的（本工作的贡献）：**
1. **Hierarchical reasoning library 用于数学推理 + 与 RL 训练 co-evolve**：SkillRL 在 agent 场景做了 co-evolution，但没有人在 math reasoning 上做
2. **Strategy + caution 的统一 library**：从成功 trace 提取正面策略，从失败 trace 提取抽象警示（非具体错题，避免 leakage），两者共存于同一个 hierarchical library
3. **小模型做题 + 大模型分析的分工**：abstraction 基于小模型的真实犯错模式量身提取
4. **逐题提取 → 跨题聚合的两步流程**：general 知识不是人工定义或大模型一次性总结的，而是从大量逐题 abstraction 中通过频率统计自然涌现的

---

## 3. 方法设计

### 3.1 Overview: Co-Evolution Loop

```
┌─────────────────────────────────────────────────────┐
│                   Full Pipeline                      │
│                                                      │
│  Cold Start:                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Base model│───→│ 做题收集  │───→│ Step 1+2:│       │
│  │          │    │ trace    │    │ 提取+聚合 │       │
│  └──────────┘    └──────────┘    └─────┬────┘       │
│                                        ↓             │
│                                  ┌──────────┐        │
│                                  │Library v1 │        │
│                                  └─────┬────┘        │
│                                        ↓             │
│                                  ┌──────────┐        │
│                                  │ Cold-Start│        │
│                                  │ SFT      │        │
│                                  └─────┬────┘        │
│                                        ↓             │
│  Co-Evolution Loop:              ┌──────────┐        │
│  ┌──────────┐                    │ Model v0  │        │
│  │ Model v_t │←───── loop ───────└──────────┘        │
│  │          │                                        │
│  │  做题 → 提取 → Library v_{t+1}                    │
│  │  GRPO w/ Library → Model v_{t+1}                  │
│  │  (validation-triggered update)                    │
│  └──────────┘                                        │
└─────────────────────────────────────────────────────┘
```

### 3.2 关于 Library 条目的设计：为什么不用 "错题本"

一个自然的想法是像学生一样维护一个 "错题本"，记录具体的错误题目和错误解题过程。但在 RL 训练的 co-evolution 设定下，这有严重的 **leakage 风险**：

- 训练集的题目在每轮 RL 中反复出现
- 如果 library 记录了 "你在第三步算错了，正确做法是..."，模型检索到后等于直接拿到了解题路径
- 模型不是学会了通用策略，而是 "记住了这道题怎么做"

**解决方案**：不记录具体的错题和解题过程，而是将经验**抽象为脱离具体题目的提醒**。Library 中的每条 abstraction 只有 1-2 句话，分为两类：

| 类型 | 来源 | 示例 | Leakage 风险 |
|---|---|---|---|
| **Strategy** | 从成功 trace 提取 | "用 Euler 定理简化大指数模运算" | 无（不含具体题目信息） |
| **Caution** | 从失败 trace 提取 | "不要在非素数模下使用 Fermat 小定理" | 无（只保留抽象教训） |

Caution 条目保留了 "从错误中学习" 的价值（strategy 说 "该怎么做"，caution 说 "别怎么做"），但丢掉了有 leakage 风险的具体信息。这本质上就是 RLAD 中 "caution alert" 类型的 abstraction。

### 3.3 Phase A: Trace 收集

- **模型**：当前轮次的小模型 $M_t$（初始为 base model）
- **数据**：训练集（AMC + MATH train，混合难度）
- **采样**：每道题用小模型采样 N 条 trace（N=8-16）
- **分类**：按最终答案正确性分为成功 trace 和失败 trace
- **关键**：每轮都重新采样——因为 $M_t$ 的能力和犯错模式与 $M_{t-1}$ 不同

### 3.4 Phase B: Library 构建（两步流程）

#### Step 1: 逐题提取（Per-Problem Extraction）

每道题独立调用一次大模型 API，输入该题的题目 + 成功/失败 trace，输出 1-3 条抽象提醒。

**对于有成功 trace 的题目**，提取 strategy 条目：
```
以下是一道数学题和一个正确的解题过程。
请总结解题过程中使用的关键策略或方法，
提炼成一条简短的、可复用的策略提醒（1-2句话）。
要求：
- 不要引用题目中的具体数字或答案
- 提醒应该能泛化到其他类似题目

题目：[problem]
正确解题过程：[successful trace]
```

**对于有失败 trace 的题目**，提取 caution 条目：
```
以下是一道数学题、一个错误的解题过程、以及正确答案。
请分析错误原因，提炼成一条简短的警示提醒（1-2句话）。
要求：
- 不要引用题目中的具体数字或答案
- 提醒应该是通用的，能帮助避免其他类似题目中的类似错误

题目：[problem]
错误解题过程：[failed trace]
正确答案：[ground truth]
```

**规模估算**：
- MATH train ≈ 7500 道题，每道调一次 o4-mini
- 每次输入约 2-3k token，输出约 100-200 token
- 总成本约 $20-50，耗时约 2-3 小时（并行调用）
- 产出：约 10000-20000 条 raw abstraction

**每条 raw abstraction 包含**（借鉴 SkillRL 的三元组格式）：
- `name`: 简洁的标识名（如 "check_coprimality"）
- `type`: strategy 或 caution
- `principle`: 1-2 句话的抽象提醒
- `when_to_apply`: 明确的使用条件（如 "当题目涉及模运算且模数非素数时"）
- `source_problem_id`: 来源题目 ID（用于去重和分析，不进入 prompt）
- `source_difficulty`: 来源题目难度
- `domain`: 数学领域标签（代数/几何/组合/数论，由大模型判断）

#### Step 2: 跨题聚合（Cross-Problem Aggregation）

逐题提取后会得到大量高度重复的 raw abstraction。跨题聚合的目标是去重、合并、分配层级。

**2a. Embedding 聚类去重**

```
题 A → [caution] "别忘了检查分母是否为零"
题 B → [caution] "分母可能为零的情况要排除"
题 C → [caution] "注意分母为零时解不成立"
    → 三条 embedding 相似度 > 0.85
    → 合并为一条："注意分母为零的情况需要排除"
    → hit_count = 3, domains = {algebra, calculus}
```

```
题 D → [strategy] "用 Euler 定理处理指数取模"
题 E → [strategy] "大指数取模可以用 Euler 定理降幂"
    → 合并为一条："用 Euler 定理简化大指数的模运算"
    → hit_count = 2, domains = {number_theory}
```

工具：BGE-M3 编码 + 层次聚类（阈值 > 0.85 合并），合并后的条目取代表性文本或让大模型重新改写。

**2b. 质量过滤**

- Answer leakage 检测：过滤包含具体数值的条目（用规则 + LLM judge）
- Trivial 过滤：过滤 "仔细审题"、"检查计算" 等无信息量条目
- 过短/过长过滤：保留 15-80 字的条目

**2c. 层级分配**

层级由 **跨域频率** 和 **跨题频率** 自然决定——general 知识不是人工定义的，而是从大量逐题 abstraction 中涌现的。

| 层级 | 分配规则 | 预期特征 |
|---|---|---|
| **General** | `hit_count ≥ 10` 或出现在 `≥ 3` 个 domain | 跨领域通用提醒。预期主要来自简单题（简单题成功率高，产出更多可靠 abstraction） |
| **Domain-specific** | `hit_count ≥ 3` 且集中在 1-2 个 domain | 领域内技巧/警示。混合难度来源 |
| **Problem-specific** | `hit_count < 3` | 具体方法。可能来自任何难度 |

**2d. 构建 FAISS 索引**

每条 abstraction 用 BGE-M3 编码，按层级分别建索引。

**最终 library 结构**：
```
Library (预计 500-2000 条去重后)
├── General (预计 30-100 条)
│   ├── [strategy] name: "check_boundary"
│   │   principle: "遇到取模问题先检查互素条件"
│   │   when_to_apply: "题目涉及模逆运算时"
│   ├── [caution] name: "zero_denominator"
│   │   principle: "注意分母为零的情况需要排除"
│   │   when_to_apply: "当解方程得到的根可能使分母为零时"
│   └── ...
├── Domain-specific (预计 100-500 条)
│   ├── Number Theory
│   │   ├── [strategy] name: "euler_theorem_modexp"
│   │   │   principle: "用 Euler 定理简化大指数模运算"
│   │   │   when_to_apply: "需要计算 a^n mod m 且 n 很大时"
│   │   ├── [caution] name: "fermat_non_prime"
│   │   │   principle: "在非素数模下误用 Fermat 小定理是常见错误"
│   │   │   when_to_apply: "当模数不是素数时，不能直接用 Fermat 小定理"
│   │   └── ...
│   ├── Algebra / Geometry / Combinatorics
│   └── ...
└── Problem-specific (预计 300-1500 条)
    ├── [strategy] name: "gcd_transform"
    │   principle: "类似题中将问题转化为求 gcd 后简化"
    │   when_to_apply: "当题目结构可以转化为 gcd 求解时"
    └── ...
```

### 3.5 Phase C: Library-Augmented GRPO 训练

#### 检索策略

对 GRPO 的每个 rollout，在生成 response 前做 hierarchical retrieval：

1. **General 层**：取 top-$k_1$ 最相关条目（$k_1$=2）
2. **Domain 层**：判断问题所属 domain，在对应 domain 内取 top-$k_2$（$k_2$=2）
3. **Problem-specific 层**：在全部 problem-specific 中取 top-$k_3$（$k_3$=2）

#### Prompt 组织

```
## General Strategies & Cautions
[strategy: check_boundary] 遇到取模问题先检查互素条件
  → Apply when: 题目涉及模逆运算时
[caution: zero_denominator] 注意分母为零的情况需要排除
  → Apply when: 当解方程得到的根可能使分母为零时

## Number Theory Strategies & Cautions
[strategy: euler_theorem_modexp] 用 Euler 定理简化大指数模运算
  → Apply when: 需要计算 a^n mod m 且 n 很大时
[caution: fermat_non_prime] 在非素数模下误用 Fermat 小定理是常见错误
  → Apply when: 当模数不是素数时

## Similar Problem Experiences
[strategy: gcd_transform] 类似题中将问题转化为求 gcd 后简化
  → Apply when: 当题目结构可以转化为 gcd 求解时
[caution: multi_solution] 类似题型中容易忽略多解情况
  → Apply when: 当方程可能有多个根时

## Problem
[x]

Please solve this problem step by step.
You may find the above strategies and cautions helpful.
```

#### GRPO 训练细节

- **Reward**：标准 verifiable reward（answer correctness）
- **Library 注入**：每个 rollout 的 prompt 中注入检索到的 abstraction
- **No-library rollout（可选）**：一部分 rollout 不注入 library（类似 RLAD 的 reward masking），防止模型过度依赖 library
- **每轮训练**：1-2 epoch over training set

### 3.6 Library 迭代更新（Round 2+）

每轮 RL 训练结束后，用更新后的模型重新收集 trace，重新走 Step 1 → Step 2。

与 Round 1 的区别在于可以**对比新旧 trace 做增量更新**：

```
新旧 trace 对比：
├── 之前做错 → 现在做对：
│   → 模型克服了某个弱点
│   → 对应的 caution 条目降权（不立刻删除，防止 regression）
│   → 可能产出新的 strategy 条目（模型学会了什么新方法）
│
├── 之前做对 → 现在做错：
│   → RL 训练可能引入了新问题
│   → 大模型分析新错误 → 新的 caution 条目
│
├── 一直做错：旧的 caution 仍然有效 → 保留
│
├── 一直做对：旧的 strategy 仍然有效 → 保留
│
└── 全局：重新做聚类去重和层级分配（hit_count 会变化）
```

**预期的 library 演化趋势**：
- 早期轮次：大量 caution（模型弱，犯错多）；strategy 偏基础
- 后期轮次：caution 减少（旧错误被克服）；strategy 更高级
- General 层趋于稳定（基础知识不会过时）；Problem-specific 层变动最大

---

## 4. Design Choices 与理由

### 4.1 为什么用统一的 Strategy/Caution 而非 "错题本 + 总结本"？

原始的 "错题本" 思路是记录具体的错误题目和错误过程。但在 co-evolution 的 RL 训练中，训练集的题目在每轮反复出现，如果 library 包含 "这道题你在第三步算错了" 这样的具体信息，模型在后续轮次检索到后等于拿到了解题路径——这是 **data leakage**。

解决方案：将所有经验**抽象为脱离具体题目的提醒**。从失败 trace 中提取的不是 "错题记录"，而是 "抽象警示"（caution）。这保留了从错误中学习的价值，同时消除了 leakage 风险。

Strategy 和 caution 共享相同的格式（1-2 句话的抽象提醒）、相同的 library 结构、相同的检索机制。唯一的区别是 framing：strategy = "该怎么做"，caution = "别怎么做"。

### 4.2 为什么用两步提取（逐题 → 聚合）而非一步总结？

| 方案 | 做法 | 问题 |
|---|---|---|
| 直接让大模型总结所有题 | "请总结做数学题的通用策略" | 太泛、不 grounded；无法知道哪些策略真的帮过模型 |
| **逐题提取 → 聚合** | 每道题独立提取 → 频率统计决定层级 | 每条 abstraction 有明确的来源证据；general 知识是涌现的不是人工定义的 |

逐题提取的额外好处：
- 可以精确追踪每条 abstraction 被多少道题触发（hit_count）
- 可以分析不同难度的题贡献了什么类型的 abstraction
- Co-evolution 中可以做增量更新（只对新变化的题重新提取）

### 4.3 为什么小模型做题 + 大模型分析？

| 方案 | 优点 | 缺点 |
|---|---|---|
| 大模型做题 + 大模型分析 | 简单 | Caution 不针对小模型弱点——大模型不犯小模型的错 |
| 小模型做题 + 小模型分析 | Self-contained | 小模型 meta-cognition 能力差，提取质量低 |
| **小模型做题 + 大模型分析** | Caution 针对小模型真实弱点；分析质量高 | 需要大模型 API（仅 offline） |

### 4.4 为什么 Co-Evolution？

Static library 反映的是 base model 的弱点。RL 训练后模型变了：旧错误被克服，新错误出现。Static library 还在提醒已经不存在的问题，浪费 prompt 空间，甚至可能 mislead。Co-evolving library 自动追踪模型成长。

### 4.5 关于 Easy-to-Hard

不强制只从简单题提取。但预期简单题自然贡献更多 general 层条目：
- 简单题成功率高 → 产出更多 strategy
- 简单题的策略更基础 → 更容易跨 domain 出现 → hit_count 高 → 被分到 General 层
- 困难题成功率低 → 产出更多 caution → 更 domain-specific

这是可验证的涌现现象，不是设计约束。

---

## 5. 与 SkillRL 的深度对比：借鉴与差异

SkillRL (Xia et al., 2026.02) 是与本工作最接近的已有工作。它在 agent 场景中实现了 hierarchical skill library + co-evolution 的完整 pipeline。理解它的设计对 positioning 本工作至关重要。

### 5.1 SkillRL 的核心设计

**Experience-based Skill Distillation**：用 teacher model 处理成功/失败轨迹。成功轨迹 → 提取 strategic pattern（关键决策点、可泛化行为）；失败轨迹 → 合成 failure lesson（失败点、flawed reasoning、正确的 counterfactual、通用防范原则）。实现 10-20× token 压缩。

**Hierarchical SkillBank**：两层结构。General Skills = 跨任务通用启发式（exploration、verification 策略）；Task-Specific Skills = 针对特定任务类型的程序性指南（preconditions、unique failure modes）。

**Skill 格式**：每个 skill 是三元组 `(name, principle, when_to_apply)`——比纯自然语言更结构化，便于检索和理解。

**检索机制**：General skills 总是被注入 prompt；Task-specific skills 通过 embedding similarity 自适应 top-K 检索。

**Recursive Evolution**：RL 训练中设 validation checkpoint。当 validation failure rate 超过阈值时触发 targeted skill refinement，确保 SkillBank 与 policy frontier 同步。有 "最低成功率阈值" 和 "每轮最大新增数" 等超参数控制更新节奏。

**Cold-Start SFT**：在进入 RL 训练前，先做一轮 supervised fine-tuning，教 policy 如何阅读和利用 skill——否则 base model 不知道怎么用 SkillBank 里的内容。

**实验**：ALFWorld（embodied manipulation）、WebShop（web navigation）、7 个 search-augmented QA 任务。比 strong baseline 提升 15.3%，ALFWorld 达到约 90% 成功率。

### 5.2 关键差异

| 维度 | SkillRL | 本工作 |
|---|---|---|
| **Domain** | Agent 任务（ALFWorld、WebShop、search QA） | 数学推理（AIME、MATH、DeepScaleR） |
| **Skill 内容** | Action pattern、procedural guide for sequential decision | Strategy/caution 提醒，针对 mathematical reasoning |
| **层级数** | 2 层（General + Task-Specific），类型预定义 | 3 层（General + Domain + Problem-Specific），基于频率统计涌现 |
| **层级分配** | 按 skill 性质人工归类 | **逐题提取 → 跨题聚合 → 频率统计自动分配** |
| **提取流程** | Teacher model per-trajectory 提取后直接分类 | **两步流程**：逐题提取 → 跨题聚类去重聚合 → 频率决定层级 |
| **Leakage 问题** | Agent 场景无此问题（每个 task 独立 session） | Math RL 中题目跨 epoch 重复，必须抽象化避免 leakage |
| **Evolution 触发** | Validation failure 触发 targeted refinement | 每轮 RL 结束后增量更新 |
| **失败经验处理** | Teacher 合成完整的 failure lesson（含 counterfactual） | 仅提取抽象 caution（脱离具体题目，防 leakage） |

### 5.3 从 SkillRL 借鉴的设计

以下设计已被 SkillRL 验证有效，本工作可以直接采用：

**1. Cold-Start SFT 阶段**

SkillRL 在 RL 前先做一轮 SFT，教模型如何使用 skill。本工作也应加入这一步：
- 构建 library v1 后，用大模型生成一批 "带 library 辅助的正确解题 trace"
- 用这些 trace 做少量 SFT，教小模型 "如何阅读 prompt 中的 strategy/caution 并在解题中运用"
- 然后再进入 GRPO 训练

不做 cold-start SFT 的风险：base model 可能完全忽略 prompt 中注入的 library 内容，GRPO 前期 rollout 质量差。

**2. Skill 三元组格式**

SkillRL 的 `(name, principle, when_to_apply)` 格式比纯自然语言更好：
- `name`：便于检索索引和日志追踪
- `principle`：核心内容（1-2 句话）
- `when_to_apply`：明确的使用条件，减少 irrelevant retrieval

本工作的 abstraction 条目可以扩展为：
```
{
  "name": "check_coprimality",
  "type": "caution",
  "principle": "不要在非互素的模下使用 Fermat 小定理",
  "when_to_apply": "当题目涉及模运算且模数非素数时",
  "level": "domain",
  "domain": "number_theory",
  "hit_count": 7
}
```

**3. Validation-Triggered Evolution**

SkillRL 不是每个训练 step 都更新 library，而是在 validation 性能停滞或下降时才触发。本工作可以借鉴：
- 每轮 RL 训练后在 held-out validation set 上评测
- 如果性能提升正常 → 不更新 library（节省大模型 API 成本）
- 如果性能停滞或下降 → 触发 library 更新（新一轮 trace 收集 + 提取）
- 这比 "每轮固定更新" 更高效

**4. Evolution 超参数**

SkillRL 已经调过的超参数可以作为起点：
- 最低成功率阈值（低于此值的 skill 被更新）
- 每轮最大新增 skill 数（防止 library 膨胀）

### 5.4 本工作相对于 SkillRL 的独特贡献

1. **Domain 迁移到 mathematical reasoning**：这不是简单的 "换个 benchmark"。Reasoning 的 abstraction 本质不同于 agent 的 action pattern——它是 procedural/factual/cautionary 的数学知识，需要不同的提取方式和质量标准
2. **三层 hierarchy + 涌现式分配**：SkillRL 的两层是预定义的；本工作新增 Domain 层（数学领域划分），且层级不是人工指定而是通过跨题频率统计自动涌现
3. **逐题 → 聚合的两步流程**：SkillRL 直接 per-trajectory 提取并分类。本工作先逐题独立提取，再跨题聚类去重、频率统计、自动分配层级——这是一个更 principled 的方法论
4. **Leakage-aware 设计**：这是 reasoning domain 特有的挑战。Agent 任务中每个 task 是独立 session，但数学 RL 训练中题目跨 epoch 重复。本工作的 strategy/caution 抽象化设计专门解决此问题
5. **Easy-to-hard 涌现分析**：SkillRL 不涉及难度梯度分析。本工作可以分析 library 各层级与题目难度的关系——General 层是否主要来自简单题？

---

## 6. 渐进式实验设计

本工作采用 **由简到繁的 staged 实验路线**。每个 stage 独立验证一个核心 claim，任何一步停下来都有可发表的结论。前 4 个 stage 完全不需要 RL 训练，3-4 周即可出结果。

### 数据集

| 角色 | 数据集 | 用途 |
|---|---|---|
| Library 构建（trace 来源） | MATH train (level 1-5) | 小模型做题 + 大模型逐题提取 |
| 评测（难题） | AIME 2024, AIME 2025, MATH test (level 4-5), DeepScaleR Hard | 评估 library 效果 |
| 评测（简单题控制组） | MATH test (level 1-3) | 验证 library 不引入噪音 |
| RL 训练集（Stage 5-6） | MATH train (level 1-5) | GRPO 训练 |
| Validation（Stage 5-6） | Held-out subset of MATH train | 触发 library 更新 |

### 模型

| 角色 | 模型 |
|---|---|
| 小模型（做题 + 训练 + 使用 library） | Qwen-3-4B（主实验）；Qwen-3-1.7B（与 RLAD 对比） |
| 大模型（逐题分析 trace、提取 abstraction） | o4-mini（仅 offline） |

---

### Stage 0：Library 本身有没有用？（Inference-Only Baseline）

**目标**：验证从小模型自身解题经验中提取的 abstraction library 能否通过检索提升 inference 性能。

**做法**：完全不做 RL、不做 hierarchy、不做 co-evolution。
1. Qwen-3-4B 在 MATH train 上跑一遍，每题采样 8 条 trace
2. o4-mini 逐题提取 strategy/caution（三元组格式）
3. 跨题聚类去重，构建一个 **flat** library
4. 在评测集上 embedding 检索 top-6 注入 prompt，直接 inference

**对比**：

| 方法 | 描述 |
|---|---|
| Vanilla CoT | 无 library，直接 chain-of-thought |
| **Flat library top-6 (ours)** | 检索 6 条最相关 abstraction 注入 prompt |

**验证的 claim**：Library 本身有用——从解题经验中提取的 abstraction 能通过检索提升推理性能。

**为什么先做这个**：如果连这一步都不 work，后面所有东西都没有基础。完全不需要 RL 训练，几天就能出结果。

**预估时间**：1-2 周

---

### Stage 1：涌现式 Hierarchy 有没有用？

**目标**：验证逐题提取 → 跨题聚合 → 频率统计的涌现式层级分配 + 分层检索优于 flat 检索。

**做法**：在 Stage 0 的 flat library 基础上，加入层级分配。
1. 对所有 abstraction 统计 hit_count 和 domain 分布
2. 按频率自动分为 General / Domain / Problem-specific 三层
3. 改用 hierarchical retrieval（每层 top-2，总计 6 条）

**对比**：

| 方法 | 描述 |
|---|---|
| Flat top-6 (Stage 0) | 所有 abstraction 混一起取 top-6 |
| **Hierarchical 每层 top-2 (ours)** | General 2 + Domain 2 + Problem-specific 2 |
| Single best | 只取 1 条最相关 |

**验证的 claim**：涌现式层级分配 + 分层检索 > flat 检索。

**附带分析（验证 easy-to-hard 涌现）**：
- General 层的条目主要来自什么难度的题？
- 三个层级的 strategy/caution 比例有什么差异？
- 去掉 easy-source / hard-source abstraction 后各层级受什么影响？

**预估时间**：额外 3-5 天

---

### Stage 2：Caution 有没有独立价值？

**目标**：验证从失败 trace 中提取的 caution 是 library 不可或缺的成分。

**做法**：在 Stage 1 的 hierarchical library 上做 ablation，无需额外 pipeline 开发。

**对比**：

| 方法 | 描述 |
|---|---|
| Strategy + Caution (Stage 1) | 完整 library |
| Strategy only | 去掉所有 caution 条目 |
| Caution only | 去掉所有 strategy 条目 |

**验证的 claim**：Caution 有独立价值；Strategy + Caution > Strategy only。

**预估时间**：额外 2-3 天

---

### Stage 3：小模型做题 + 大模型分析 vs 其他方案

**目标**：验证 "针对小模型弱点量身提取" 的 library 比其他提取方式更有效。

**做法**：构建三个不同来源的 library，都用同样的 hierarchical retrieval 评测。

**对比**：

| 方法 | 做题 | 分析 |
|---|---|---|
| **(a) Ours** | 小模型 (Qwen-3-4B) | 大模型 (o4-mini) |
| (b) 大模型全包 | 大模型 (o4-mini) | 大模型 (o4-mini) |
| (c) 小模型全包 | 小模型 (Qwen-3-4B) | 小模型 (Qwen-3-4B) |

**验证的 claim**：(a) 最优——caution 针对小模型真实弱点；(b) 的 caution 不匹配小模型；(c) 分析质量差。

**预估时间**：额外 3-5 天

---

> **Stage 0-3 Checkpoint**：到此为止全部是 inference-time 实验，总计约 3-4 周。如果结果正面，已经有四个独立的 claim 可以撑起一篇 workshop paper 或完整论文的前半部分。

---

### Stage 4：Cold-Start SFT 有没有必要？

**目标**：验证先教模型 "如何使用 library" 再让它用，效果是否更好。

**做法**：
1. 用 Stage 1 的 library，让 o4-mini 生成一批 "带 library 辅助的正确解题 trace"
2. 用这些 trace 对 Qwen-3-4B 做少量 SFT → model v0
3. model v0 在评测集上 inference（with library）

**对比**：

| 方法 | 描述 |
|---|---|
| 直接注入 (Stage 1) | Base model + library in prompt，直接 inference |
| **Cold-start SFT (ours)** | SFT 教模型用 library → 再 inference with library |

**验证的 claim**：SFT 让模型学会利用 library 后，效果比直接塞 prompt 更好。

**预估时间**：额外 1 周

---

### Stage 5：Library-Augmented GRPO（Static Library）

**目标**：验证 library 辅助的 RL 训练优于无 library 的 RL。

**做法**：
1. 从 Stage 4 的 SFT 模型出发
2. 做标准 GRPO 训练，每个 rollout 注入 library（hierarchical retrieval）
3. 标准 verifiable reward（answer correctness）
4. Library 保持 static（不更新）

**对比**：

| 方法 | 描述 |
|---|---|
| Vanilla GRPO | 无 library，从 base model 训练 |
| SFT-then-GRPO (no library) | Cold-start SFT 但 GRPO 时不注入 library |
| **Library-augmented GRPO (ours)** | Cold-start SFT + GRPO with static library |

**验证的 claim**：Library 辅助的 GRPO > 无 library 的 GRPO。

**可加入的对照**：一部分 rollout 不注入 library（no-library rollout + reward masking），观察是否防止模型过度依赖 library。

**预估时间**：额外 2 周

---

### Stage 6：Co-Evolution

**目标**：验证 library 与模型 co-evolve 优于 static library。

**做法**：
1. Stage 5 的 GRPO 训练若干 epoch 后，用更新后的模型重新收集 trace
2. o4-mini 重新逐题提取 → 跨题聚合 → library v2
3. 继续 GRPO with library v2
4. 重复（validation-triggered：性能停滞时才更新 library）
5. 总共 3-5 轮 co-evolution

**对比**：

| 方法 | 描述 |
|---|---|
| Static library + GRPO (Stage 5) | Library 固定不变 |
| **Co-evolving library + GRPO (ours)** | Library 随模型进步更新 |

**验证的 claim**：Co-evolving > static——动态更新的 library 追踪模型成长，保持相关性。

**分析实验（Library Evolution Analysis）**：
- Library 规模随轮次如何变化？
- 哪些 abstraction 最 "长寿"？（跨多轮有用 = 真正 general 的知识）
- 哪些最 "短命"？（只在某轮有用 = 模型特定阶段的弱点）
- Library 是否收敛？
- Strategy/caution 比例如何随训练演化？
- 旧 caution 被克服后新 caution 出现的模式

**预估时间**：额外 2 周

---

### Stage 7（Optional）：Weak-to-Strong + 无监督 RL Extension

**Exp 7a: Weak-to-strong**
- 用 Qwen-3-4B 做题 + o4-mini 分析构建的 library → 提供给更强模型使用
- 小模型的经验 library 是否对强模型也有参考价值？

**Exp 7b: Library-Augmented Unsupervised RL**

| 条件 | 训练方式 | Library |
|---|---|---|
| GRPO + library (upper bound) | 有标签 GRPO | Co-evolving |
| TTRL alone | 无标签 TTRL | 无 |
| **TTRL + library** | 无标签 TTRL | Co-evolving |
| Intuitor alone | 无监督 Intuitor | 无 |
| **Intuitor + library** | 无监督 Intuitor | Co-evolving |

核心问题：TTRL + library 能否接近 GRPO alone？如果是，说明 library 可以部分替代 ground-truth labels。

---

### 实验路线总结

| Stage | 需要什么 | 新增工作量 | 验证的 claim | 累计时间 |
|---|---|---|---|---|
| **0** | Inference only | 1-2 周 | Library 本身有用 | ~2 周 |
| **1** | +层级分配 | +3-5 天 | Hierarchy > flat；easy-to-hard 涌现 | ~2.5 周 |
| **2** | +ablation | +2-3 天 | Caution 有独立价值 | ~3 周 |
| **3** | +对比实验 | +3-5 天 | 小模型做题+大模型分析最优 | ~3.5 周 |
| — | **Checkpoint** | — | **Stage 0-3 足以发 workshop paper** | — |
| **4** | +少量 SFT | +1 周 | Cold-start SFT 有必要 | ~4.5 周 |
| **5** | +GRPO | +2 周 | Library-augmented RL > vanilla RL | ~6.5 周 |
| **6** | +迭代更新 | +2 周 | Co-evolution > static | ~8.5 周 |
| **7** | +optional | +2-3 周 | Weak-to-strong; 无监督 RL + library | ~11 周 |

**最小可发表单元**：Stage 0-3（~3.5 周）= inference-time hierarchical library 的完整验证。
**完整论文**：Stage 0-6（~8.5 周）= 包含 co-evolution 的完整 pipeline。
**Extended version**：Stage 0-7（~11 周）= 包含 weak-to-strong 和无监督 RL extension。

---

## 7. 预期贡献总结

1. **Hierarchical Reasoning Library**：首次在数学推理任务上构建 general → domain → problem-specific 三层 abstraction library，层级通过跨题频率统计自然涌现
2. **Strategy + Caution 的统一设计**：从成功和失败经验中分别提取正面策略和抽象警示，避免 leakage 风险，证明 caution 的独立价值
3. **逐题提取 → 跨题聚合**：general 知识不是人工定义的而是涌现的，每条 abstraction 有明确来源证据
4. **"小模型做题 + 大模型分析" 范式**：让 abstraction 针对目标模型真实弱点量身定制
5. **Co-Evolving Library**（Stage 5-6）：首次在数学推理 RL 训练中实现 library 与模型共同进化，证明动态更新优于 static
6. **Library Evolution 分析**：揭示模型在 RL 训练中的成长轨迹——知识的产生、稳定与淘汰

---

## 8. 风险与 Mitigation

| 风险 | 可能性 | 最早暴露于 | Mitigation |
|---|---|---|---|
| Library 本身没用（检索 = noise） | 中 | Stage 0 | 如果 Stage 0 fail，尝试更严格的质量过滤或更好的检索方式；worst case 2 周内知道 |
| Hierarchical 不优于 flat | 中 | Stage 1 | 分析性结论（层级分布、难度来源）仍有价值 |
| Caution 引入 noise > signal | 中 | Stage 2 | 质量过滤；比例可调；strategy-only 仍然 work |
| 小模型分析不行、大模型分析也差不多 | 低-中 | Stage 3 | 至少证明了 "谁做题" 这个维度的影响 |
| Cold-start SFT 没有明显提升 | 中 | Stage 4 | 省掉这一步，直接进 GRPO |
| Library-augmented GRPO 不优于 vanilla | 中 | Stage 5 | Stage 0-3 的 inference-time 结论仍然成立 |
| Co-evolution 不优于 static | 中 | Stage 6 | Static library 本身已是贡献 |
| 大模型 API 成本 | 低-中 | 全程 | 用 o4-mini；每轮约 $20-50 |
| Abstraction leakage | 低 | Stage 0 | 提取 prompt 要求不含数字；LLM judge 过滤 |

**关键风险策略**：前 4 个 stage 总共只需 3-4 周且不需要 RL，可以快速验证基本假设是否成立。如果 Stage 0 就 fail，及时止损；如果 Stage 0-3 成功，即使后续 RL 相关 stage 效果不如预期，已有足够的 inference-time 结果可以发表。

---

## 9. 竞争态势

| 团队 | 已有工作 | 与本工作重叠 | 威胁 |
|---|---|---|---|
| Aviral Kumar / Chelsea Finn | RLAD (2025.10) | RL 内 abstraction，但不做外部 library | 中 |
| Didolkar / Bengio (Meta) | Metacognitive Reuse (2025.09) | 可能做 evolving handbook，但关注 efficiency | 中-高 |
| Suzgun / Zou (Stanford) | Dynamic Cheatsheet (2025.04) | Test-time，不涉及 RL 训练 | 低 |
| SkillRL 团队 | SkillRL (2026.02) | 最接近（hierarchical + co-evolution），但 agent 场景 | 中 |
| TTRL 团队 (Tsinghua) | TTRL (2025.04) | 无监督 RL，不涉及 library | 低 |

**窗口期约 3-6 个月。**

---

## 10. 核心参考文献

1. **RLAD** — Qu et al., arXiv:2510.02263, Oct 2025
2. **Metacognitive Reuse** — Didolkar et al., arXiv:2509.13237, Sep 2025
3. **Dynamic Cheatsheet** — Suzgun et al., EACL 2026
4. **ExpeL** — Zhao et al., AAAI 2024
5. **SkillRL** — Xia et al., arXiv:2602.08234, Feb 2026
6. **LeMa** — An et al., arXiv:2310.20689, 2024
7. **Can LLMs Learn from Mistakes?** — Tong et al., ACL 2024
8. **SKiC** — Chen et al., ACL Findings 2024
9. **E2H Reasoner** — arXiv:2506.06632, Jun 2025
10. **H-MEM** — arXiv:2507.22925, Jul 2025
11. **TTRL** — Zuo et al., NeurIPS 2025
12. **Intuitor** — Zhao et al., ICLR 2026
13. **Programming by Backprop** — Cook et al., arXiv:2506.18777, Jun 2025
14. **Mistake Notebook Learning** — arXiv:2512.11485, Dec 2025
