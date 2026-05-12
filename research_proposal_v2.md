# HRLib v2: Library as Training Signal — Updated Plan Post Stage 0

## 1. What Stage 0 Told Us

### The Experiment
We built a ~9k-entry reasoning abstraction library (strategy + caution bullets) from Qwen-3-1.7B-base's own traces on MATH-train, then injected retrieved bullets into prompts at inference time on MATH-500.

### Key Numbers

| Metric | Vanilla (no library) | Best library injection |
|---|---|---|
| pass@1 | 0.5188 | 0.5296 (+1.1pp) |
| pass@32 | 0.8700 | 0.8780 (+0.8pp) |
| Net accuracy lift | — | **~zero** |

But the library mechanism itself works:
- 79-86% of problems have ≥1 relevant bullet retrieved
- When relevant, 68.8% of bullets are used correctly in correct rollouts
- Caution bullets outperform strategy on relevance (36% vs 27%) and lower misuse (1.4% vs 2.2%)
- Model is robust to noise: 99.5% of irrelevant bullets cleanly ignored

### The Problem
- 73-78% of relevant bullets are **ignored** even in correct rollouts
- The model doesn't know how to use library content — it's a base model with no instruction-following training
- Retrieval improvements (query rewriting, cross-encoder reranking) raised relevance from 29% → 34% but **did not improve accuracy**
- Hierarchy and strategy/caution ablations were not promising

### The Conclusion
**Inference-time injection alone does not work for untrained base models.** The bottleneck is not retrieval quality — it's that the model has never learned the behavior pattern "read strategy hints in prompt → apply them in reasoning." This is exactly what SkillRL found: removing cold-start SFT caused a 20% performance drop (89.9% → 65.2%).

---

## 2. Revised Core Thesis

**Old thesis**: "Library improves reasoning via inference-time retrieval (RAG for reasoning)."

**New thesis**: "The value of an abstraction library is not as an inference-time crutch, but as a **structured training signal**. When a model is trained to utilize library content (via SFT), and then fine-tuned with library-augmented RL, the library becomes a powerful curriculum that accelerates learning and transfers knowledge from simple to complex problems."

This is a stronger and more novel claim than the original:
- It has a clear negative result (Stage 0) as motivation
- It explains *why* SFT + RL is necessary (not just "SkillRL did it too")
- It positions the library as a **training methodology**, not just retrieval augmentation
- It opens up weak-to-strong and easy-to-hard angles naturally

---

## 3. Revised Contributions

| # | Contribution | Evidence / Stage |
|---|---|---|
| 1 | **Library as training signal, not inference-time injection**: Demonstrate that abstraction libraries only become effective when the model is trained to use them. Inference-time injection alone ≈ zero lift. | Stage 0 (done) + Stage 4-5 |
| 2 | **"Small model does problems + large model analyzes" paradigm**: Abstractions are tailored to the target model's real weaknesses, not generic advice. SFT data is constructed from the small model's actual failure traces + relevant library bullets. | Stage 4 |
| 3 | **Library-augmented RL outperforms vanilla RL**: Library provides structured hints during rollouts that help the model explore better solutions. | Stage 5 |
| 4 | **Weak-to-strong structured transfer**: Large model analyzes small model's weaknesses → extracts targeted abstractions → generates SFT data showing how to use them → small model learns. This is more targeted than generic distillation. | Stage 4-5, ablation |
| 5 | **Easy-to-hard via library composition**: Abstractions from simple problems (where the model succeeds more) form the "general" layer; these transfer to help on hard problems during RL training. | Analysis experiment |
| 6 | **(If Stage 6 reached) Co-evolving library**: Library updates as model improves, tracking evolving weaknesses. | Stage 6 |

---

## 4. Staged Experiment Plan (Revised)

### Stage 4: Cold-Start SFT (THE CRITICAL NEXT STEP)

**Goal**: Teach the model to read, understand, and apply library bullets in its reasoning.

#### 4a. Construct SFT Data

**Inputs needed**:
- MATH-train problems (you have these)
- The existing library (~9k entries)
- Small model's traces from Stage 0 (success + failure traces)
- Judge data from Stage 0 (which bullets are relevant to which problems)

**Pipeline**:

```
For each training problem:
  1. Look up which library bullets are relevant (from judge data, or re-retrieve with gated pipeline)
  2. Find the small model's failure trace for this problem (if any)
  3. Send to teacher model:
     - Problem + relevant bullets + small model's failure trace
     - Ask: "Generate a correct solution that explicitly references the provided strategies/cautions by name"
  4. Verify teacher's answer is correct (using existing reward function)
  5. Keep only verified-correct examples
```

**Teacher prompt**:
```
You are helping train a small math model to use reasoning strategies effectively.

## Problem
{problem}

## Available Strategies & Cautions
{relevant_bullets_formatted}

## The Small Model's Failed Attempt
{failed_trace}

## Your Task
Solve this problem correctly. In your solution:
1. Explicitly reference relevant strategies/cautions by their [name] tags
2. Show how applying them leads to the correct approach
3. If the small model's error relates to a caution, point out how that caution would have prevented the mistake

Write your solution in a natural chain-of-thought style with <think>...</think> tags.
```

**SFT data format** (for verl):
```python
{
    "text": tokenizer.apply_chat_template([
        {"role": "system", "content": library_bullets_formatted},
        {"role": "user", "content": problem_text},
        {"role": "assistant", "content": teacher_solution_with_bullet_references}
    ], tokenize=False)
}
```

**Scale**: 1000-2000 verified-correct examples should be enough. SkillRL used 7500 for ALFWorld but that's a different domain. Math reasoning needs fewer but higher-quality examples.

**Teacher model**: DeepSeek V3.2 ($5 per batch of 2000) or R1 ($14) or free model if quality is sufficient.

**Key difference from SkillRL**: SkillRL's teacher generates solutions from scratch. Our teacher sees the **small model's actual failure** and generates a solution that shows how the relevant bullets would have prevented that failure. This makes the SFT data targeted to the small model's real weaknesses.

#### 4b. Run SFT

```bash
python -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/sft_library_aware.parquet \
    model.path=Qwen/Qwen3-1.7B \
    trainer.total_epochs=3 \
    trainer.learning_rate=1e-5 \
    trainer.save_path=checkpoints/sft_library_aware
```

#### 4c. Evaluate (THE KEY COMPARISON TABLE)

Run inference on MATH-500 under 5 conditions:

| Condition | Training | Inference prompt | What it tests |
|---|---|---|---|
| A. Base + no library | None | Bare problem | Stage 0 vanilla baseline |
| B. Base + library | None | Problem + retrieved bullets | Stage 0 result (library alone ≈ useless) |
| C. **Library-aware SFT + library** | SFT with bullet references | Problem + retrieved bullets | **Core hypothesis: SFT teaches model to use library** |
| D. Library-aware SFT + no library | SFT with bullet references | Bare problem (no bullets) | Does SFT internalize knowledge even without library at inference? |
| E. Generic SFT + library | SFT on teacher solutions WITHOUT bullet references | Problem + retrieved bullets | Is library-aware SFT better than generic distillation? |

**Expected results**:
- C >> B (SFT is necessary for library to work)
- C > D (library at inference time adds value on top of SFT)
- C > E (library-aware SFT is better than generic SFT + library)
- D > A (SFT internalizes some library knowledge)

**If C >> B and C > E**: This is the paper's core result. Library is a training signal, not an inference-time trick. And library-aware SFT (where the model learns to reference specific bullets) is better than generic distillation.

**Metrics**: pass@1, pass@4, pass@8, pass@32 on MATH-500. Also re-run LLM-as-judge to see if bullet usage rates improve dramatically under condition C vs B.

**Time estimate**: 1-2 weeks (data generation + SFT + eval)

---

### Stage 5: Library-Augmented GRPO

**Goal**: Verify that library-augmented RL training outperforms vanilla RL.

**Setup**: Start from Stage 4's SFT checkpoint. Run GRPO on MATH-train with library bullets injected into rollout prompts.

**Data preparation** (same as before — inject library into verl parquet):
```python
# For each MATH-train problem, retrieve top-6 bullets, format into prompt
# 80% of rollouts get library injection, 20% bare (no-library rollout)
```

**Comparison**:

| Condition | SFT init | GRPO rollout prompt | What it tests |
|---|---|---|---|
| F. Vanilla GRPO | Base model | Bare problem | Standard RL baseline |
| G. SFT → GRPO (no library) | Library-aware SFT | Bare problem | SFT helps RL but no library during RL |
| H. SFT → GRPO + static library | Library-aware SFT | Problem + retrieved bullets | **Library as RL training signal** |
| I. Generic SFT → GRPO + library | Generic SFT | Problem + retrieved bullets | Library-aware SFT matters for RL too? |

**Expected**: H > G > F. Library during RL rollouts helps the model explore better solutions because it has hints pointing toward correct strategies.

**Time estimate**: 2 weeks

---

### Stage 6: Co-Evolution (if Stage 5 is positive)

Same as proposal v5: after GRPO training, re-collect traces, update library, repeat.

**Key new angle**: Track whether the model's bullet usage rate (from judge) improves across rounds. In Stage 0 the model ignored 73-78% of relevant bullets. After SFT + RL + co-evolution, this should drop significantly.

---

### Stage 7 (Optional): Weak-to-Strong and Easy-to-Hard Analysis

#### 7a. Weak-to-Strong

The entire pipeline IS weak-to-strong:
```
Weak (Qwen-1.7B) does problems → exposes real weaknesses
Strong (DeepSeek/GPT) analyzes weak model's failures → extracts targeted abstractions
Strong generates SFT data showing how to use abstractions → trains weak model
Weak model becomes stronger
```

**Ablation**: Compare our pipeline (strong analyzes weak's failures) vs generic distillation (strong just solves problems, weak imitates). Our pipeline should win because the abstractions are targeted.

**Extension**: Can the library built from Qwen-1.7B's weaknesses help Qwen-4B? (Cross-model transfer)

#### 7b. Easy-to-Hard

**Analysis during Stage 5 RL training**:
- Track which library bullets are retrieved during GRPO rollouts on hard problems (MATH level 4-5)
- Of those, how many originated from simple problems (MATH level 1-3)?
- Do hard-problem rollouts that use easy-sourced bullets succeed more often?

**Ablation**: Build two libraries:
- Easy-only library (from MATH level 1-3 traces)
- Hard-only library (from MATH level 4-5 traces)
- Mixed (current)

Compare RL training with each. If easy-only library performs well on hard problems, that's a strong easy-to-hard transfer result.

---

## 5. What to Borrow from SkillRL vs Do Differently

### Borrow

| Component | From SkillRL | How we use it |
|---|---|---|
| Cold-start SFT is mandatory | Their ablation: -20% without it | Our Stage 0 confirms this independently |
| Skill format: (name, principle, when_to_apply) | Their structured format | Already using this |
| General skills always injected, task-specific retrieved | Their retrieval design | Adapt for math domains |
| Validation-triggered evolution | Their update schedule | Use in Stage 6 |
| No-library rollouts during RL | Implicit in their design | 20% bare rollouts to prevent over-reliance |

### Do Differently

| Dimension | SkillRL | Ours | Why different |
|---|---|---|---|
| Domain | Agent tasks (ALFWorld, WebShop) | Math reasoning (MATH, AIME) | Different abstraction types |
| SFT data construction | Teacher solves from scratch with skills | Teacher sees small model's failure + shows how skills prevent that failure | More targeted to weak model's actual weaknesses |
| Leakage management | Not needed (agent episodes independent) | Must abstract away specific numbers/answers | Math RL reuses same problems across epochs |
| Library construction | Per-trajectory distillation | Per-problem extraction → cross-problem aggregation | Frequency statistics reveal what's truly "general" |
| Hierarchy assignment | Predefined (general vs task-specific) | Emergent from cross-problem frequency stats | More principled, data-driven |
| Core claim | "Skills help agents" | "Library's value is as training signal, not inference-time injection" | Motivated by our Stage 0 negative result |
| Negative result | None reported | Inference-time injection ≈ zero lift on untrained model | Stronger motivation for SFT |

---

## 6. Practical Notes

### Cost Estimate

| Step | Cost |
|---|---|
| SFT data generation (2000 problems × DeepSeek V3.2) | ~$5-10 |
| SFT training (Qwen-1.7B, 3 epochs) | GPU time only |
| GRPO training | GPU time only |
| Library re-extraction per co-evolution round | ~$5 |
| **Total API cost for complete paper** | **~$30-50** |

### Hardware

| Stage | Minimum |
|---|---|
| Stage 4 (SFT) | 2×A100 |
| Stage 5 (GRPO) | 4×A100 for Qwen-1.7B |
| Stage 6 (co-evolution) | Same as Stage 5 |

### Timeline

| Stage | Time | Cumulative |
|---|---|---|
| 4a: SFT data generation | 3-4 days | 4 days |
| 4b: SFT training | 1-2 days | 6 days |
| 4c: Eval (5 conditions) | 2-3 days | 9 days |
| 5: GRPO experiments | 2 weeks | 3.5 weeks |
| 6: Co-evolution (if positive) | 2 weeks | 5.5 weeks |
| 7: Analysis + writing | 2 weeks | 7.5 weeks |

---

## 7. Success Criteria

### Minimum viable paper (Stage 4 alone)
- Show that library-aware SFT + library (condition C) significantly outperforms:
  - Base + library (condition B): proves SFT is necessary
  - Generic SFT + library (condition E): proves library-aware SFT is better than distillation
- Include Stage 0 negative result as motivation
- This alone is a focused contribution paper

### Full paper (Stage 4 + 5)
- Add GRPO results showing library-augmented RL > vanilla RL
- Include weak-to-strong analysis

### Extended paper (Stage 4 + 5 + 6)
- Co-evolution results
- Easy-to-hard analysis
- Library evolution dynamics

---

## 8. Files Needed for Stage 4 Implementation

```
scripts/
├── generate_sft_data.py           # Call teacher model with problem + bullets + failure trace
├── filter_sft_data.py             # Verify teacher answers, keep correct only
├── prepare_sft_parquet.py         # Convert to verl format
├── run_sft.sh                     # verl SFT command
├── eval_sft_conditions.py         # Run all 5 conditions (A-E) on MATH-500
data/
├── sft_library_aware.parquet      # Final SFT data for verl
├── sft_generic.parquet            # Control: SFT without bullet references
configs/
├── sft_library_aware.yaml         # verl SFT config
```