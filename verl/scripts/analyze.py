import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def per_prompt_output_accuracy(jsonl_path):
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


def per_prompt_output_diversity(jsonl_path):
    """
    Compute output diversity per prompt.
    diversity(prompt) = (# unique answers) / (# rollouts)
    """
    prompt_to_answers = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["input"]
            solution_str = data["output"]
            
            try:
                string_in_last_boxed = last_boxed_only_string(solution_str)
                if string_in_last_boxed is not None:
                    pred_ans = remove_boxed(string_in_last_boxed)
                    if prompt not in prompt_to_answers:
                        prompt_to_answers[prompt] = []
                    prompt_to_answers[prompt].append(pred_ans)
            except Exception as e:
                print(f"Error processing line: {e}")
    # Compute diversity per prompt
    prompt_to_diversity = {prompt: len(set(answers)) / len(answers) for prompt, answers in prompt_to_answers.items()}
    return prompt_to_diversity  


def to_bin(x, metric="accuracy", eps=1e-9):
    if np.isnan(x):
        return None
    if x < 0 - eps or x > 1 + eps:
        return None
    
    if metric == "accuracy":
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
    elif metric == "diversity":
        if (x > 0.0 + eps) and (x <= 0.2 + eps):
            return 0
        if (x > 0.2 + eps) and (x <= 0.4 + eps):
            return 1
        if (x > 0.4 + eps) and (x <= 0.6 + eps):
            return 2
        if (x > 0.6 + eps) and (x < 1.0 - eps):
            return 3
        if np.isclose(x, 1.0, atol=eps):
            return 4
    return None


def build_heatmap_counts(dict1, dict2, metric="accuracy"):
    keys = sorted(set(dict1.keys()) & set(dict2.keys()))
    X = np.zeros((5, 5), dtype=int)

    total = 0
    for k in keys:
        b1 = to_bin(dict1[k], metric=metric)
        b2 = to_bin(dict2[k], metric=metric)
        if b1 is None or b2 is None:
            continue
        X[b1, b2] += 1
        total += 1
    
    X = 100*X / total if total > 0 else X
    return X


def plot_acc_vs_div(acc, diversity, model_name):
    common_prompts = set(acc.keys()) & set(diversity.keys())
    acc_vals = np.array([acc[p] for p in common_prompts])
    div_vals = np.array([diversity[p] for p in common_prompts])
    print(f"{model_name} - Number of common prompts: {len(common_prompts)}")
    
    # --- Linear Regression ---
    X = div_vals.reshape(-1, 1)
    y = acc_vals
    reg = LinearRegression().fit(X, y)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(X, y)
    print("Linear Regression Fit:")
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("R^2:", r2)
    print("\n")

    x_line = np.linspace(div_vals.min(), div_vals.max(), 200).reshape(-1,1)
    y_line = reg.predict(x_line)

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(div_vals, acc_vals, alpha=0.6)
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Linear Fit (R^2={r2:.3f})")
    plt.xlabel("Output Diversity (#unique / #rollouts)")
    plt.ylabel("Accuracy")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_acc_vs_div.png", dpi=300)
    plt.close()


def plot_heatmap(dict1, dict2, model1, model2, metric="accuracy"):
    if metric == "accuracy":
        BIN_LABELS = ['0', '(0, 0.4]', '(0.4, 0.6]', '(0.6, 1)', '1']
    elif metric == "diversity":
        BIN_LABELS = ['(0, 0.2]', '(0.2, 0.4]', '(0.4, 0.6]', '(0.6, 1)', '1']
    X = build_heatmap_counts(dict1, dict2, metric=metric)

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
    fig.savefig(f"{model1}_{model2}_{metric}_heatmap.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    jsonl_path = "/raid/xinyul2/eval/base/Qwen3-4B-Base/0.jsonl"
    base_acc = per_prompt_output_accuracy(jsonl_path)
    base_diversity = per_prompt_output_diversity(jsonl_path)
    
    jsonl_path = "/raid/xinyul2/eval/grpo-naive/Qwen3-4B-Base/0307-173622/global_step_58/0.jsonl"
    gt_acc = per_prompt_output_accuracy(jsonl_path)
    gt_diversity = per_prompt_output_diversity(jsonl_path)

    jsonl_path = "/raid/xinyul2/eval/grpo-intuitor/Qwen3-4B-Base/0307-135909/global_step_58/0.jsonl"
    intuitor_acc = per_prompt_output_accuracy(jsonl_path)
    intuitor_diversity = per_prompt_output_diversity(jsonl_path)

    # check if diversity is a good proxy of accuracy
    plot_acc_vs_div(base_acc, base_diversity, "Base")
    plot_acc_vs_div(gt_acc, gt_diversity, "Ground Truth")
    plot_acc_vs_div(intuitor_acc, intuitor_diversity, "Intuitor")

    # check how output accuracy changed after training
    plot_heatmap(base_acc, gt_acc, "Base", "Ground Truth", metric="accuracy")
    plot_heatmap(base_acc, intuitor_acc, "Base", "Intuitor", metric="accuracy")
    plot_heatmap(gt_acc, intuitor_acc, "Ground Truth", "Intuitor", metric="accuracy")

    # check how output diversity changed after training
    plot_heatmap(base_diversity, gt_diversity, "Base", "Ground Truth", metric="diversity")
    plot_heatmap(base_diversity, intuitor_diversity, "Base", "Intuitor", metric="diversity")
    plot_heatmap(gt_diversity, intuitor_diversity, "Ground Truth", "Intuitor", metric="diversity")
