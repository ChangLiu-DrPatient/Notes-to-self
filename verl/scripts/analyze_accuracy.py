import json
import numpy as np
import matplotlib.pyplot as plt


def per_prompt_accuracy_analysis(jsonl_path):
    """Analyze accuracy per prompt."""
    prompt_to_scores = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["input"]
            score = data["score"]
            if prompt not in prompt_to_scores:
                prompt_to_scores[prompt] = []
            prompt_to_scores[prompt].append(score)

    # Compute average accuracy per prompt
    prompt_to_avg_score = {prompt: np.mean(scores) for prompt, scores in prompt_to_scores.items()}
    return prompt_to_avg_score


BIN_LABELS = ['0', '(0, 0.4]', '(0.4, 0.6]', '(0.6, 1)', '1']

def to_bin(x, eps=1e-9):
    if np.isnan(x):
        return None
    if x < 0 - eps or x > 1 + eps:
        return None

    if np.isclose(x, 0.0, atol=eps):
        return 0
    if (x > 0.0 + eps) and (x <= 0.4 + eps):
        return 1
    if (x > 0.4 + eps) and (x <= 0.6 + eps):
        return 2
    if (x > 0.6 + eps) and (x < 1.0 - eps):
        return 3
    if np.isclose(x, 1.0, atol=eps):
        return 4
    return None


def build_heatmap_counts(dict1, dict2):
    keys = sorted(set(dict1.keys()) & set(dict2.keys()))
    X = np.zeros((5, 5), dtype=int)

    total = 0
    for k in keys:
        b1 = to_bin(dict1[k])
        b2 = to_bin(dict2[k])
        if b1 is None or b2 is None:
            continue
        X[b1, b2] += 1
        total += 1
    
    X = 100*X / total if total > 0 else X
    return X


def plot_heatmap(dict1, dict2, model1, model2):
    X = build_heatmap_counts(dict1, dict2)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(X, cmap='Oranges', vmin=0, vmax=50)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(BIN_LABELS)
    ax.set_yticklabels(BIN_LABELS)
    
    for (i, j), val in np.ndenumerate(X):
        ax.text(j, i, f"{val:.1f}", ha='center', va='center', color='black', fontsize=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion')
    ax.set_xlabel(model2)
    ax.set_ylabel(model1)
    fig.savefig(f"{model1}_{model2}.png", dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    jsonl_path = "/raid/xinyul2/eval/base/Qwen3-4B-Base/0.jsonl"
    base_acc = per_prompt_accuracy_analysis(jsonl_path)
    
    jsonl_path = "/raid/xinyul2/eval/grpo-naive/Qwen3-4B-Base/0307-173622/global_step_58/0.jsonl"
    grpo_acc = per_prompt_accuracy_analysis(jsonl_path)

    jsonl_path = "/raid/xinyul2/eval/grpo-intuitor/Qwen3-4B-Base/0307-135909/global_step_58/0.jsonl"
    intuitor_acc = per_prompt_accuracy_analysis(jsonl_path)
    
    plot_heatmap(base_acc, grpo_acc, "Base", "GRPO")
    plot_heatmap(base_acc, intuitor_acc, "Base", "Intuitor")
    plot_heatmap(grpo_acc, intuitor_acc, "GRPO", "Intuitor")
