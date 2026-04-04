import json
import os

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


def overall_accuracy(jsonl_path):
    """
    Return (micro_acc, macro_acc, n_lines, n_prompts).

    - micro: mean of every rollout line's score (weights rollouts).
    - macro: mean of per-prompt average scores (weights each prompt equally).
    """
    prompt_to_scores = {}
    all_scores = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["input"]
            score = data["score"]
            all_scores.append(score)
            if prompt not in prompt_to_scores:
                prompt_to_scores[prompt] = []
            prompt_to_scores[prompt].append(score)
    micro = float(np.mean(all_scores)) if all_scores else float("nan")
    per_prompt = [np.mean(scores) for scores in prompt_to_scores.values()]
    macro = float(np.mean(per_prompt)) if per_prompt else float("nan")
    return micro, macro, len(all_scores), len(prompt_to_scores)


def pass_at_k_means(jsonl_path):
    """
    Mean pass@k across prompts from JSONL lines that store duplicated prompt-level metrics.

    Reads every key starting with ``pass@`` (e.g. ``pass@32``, ``pass@16``). For each prompt
    (``input``), merges ``pass@*`` from all rollouts (values are identical per prompt; first
    seen wins if keys differ). Then averages each metric over prompts that define that key.

    Returns:
        dict[str, float]: mapping ``pass@...`` -> mean value across prompts that had that key.
    """
    prompt_to_pass = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["input"]
            if prompt not in prompt_to_pass:
                prompt_to_pass[prompt] = {}
            row_pass = {k: data[k] for k in data if k.startswith("pass@")}
            for k, v in row_pass.items():
                if k not in prompt_to_pass[prompt]:
                    prompt_to_pass[prompt][k] = v

    if not prompt_to_pass:
        return {}

    all_keys = set()
    for pk in prompt_to_pass.values():
        all_keys.update(pk.keys())

    out = {}
    for key in sorted(all_keys):
        vals = []
        for pass_keys in prompt_to_pass.values():
            if key in pass_keys:
                v = pass_keys[key]
                if isinstance(v, (bool, int, float, np.number)):
                    vals.append(float(v))
        if vals:
            out[key] = float(np.mean(vals))
    return out


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


def per_prompt_output_entropy(jsonl_path):
    """
    Shannon entropy of the empirical distribution over extracted answers per prompt.

    Each distinct extracted answer is a category; p_i = count_i / n_rollouts.
    Entropy uses natural log (nats): H = -sum_i p_i log(p_i).
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
    prompt_to_entropy = {}
    for prompt, answers in prompt_to_answers.items():
        if not answers:
            continue
        n = len(answers)
        _, counts = np.unique(answers, return_counts=True)
        p = counts.astype(np.float64) / n
        h = float(-np.sum(p * np.log(p)))
        prompt_to_entropy[prompt] = h
    return prompt_to_entropy


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


def plot_acc_vs_div(acc, diversity, experiment_name, fig_dir):
    common_prompts = set(acc.keys()) & set(diversity.keys())
    acc_vals = np.array([acc[p] for p in common_prompts])
    div_vals = np.array([diversity[p] for p in common_prompts])
    print(f"{experiment_name} - Number of common prompts: {len(common_prompts)}")
    
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

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(div_vals, acc_vals, alpha=0.6)
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Linear Fit (R^2={r2:.3f})")
    plt.xlabel("Output Diversity (#unique / #rollouts)")
    plt.ylabel("Accuracy")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f"{experiment_name}_acc_vs_div.png"), dpi=300)
    plt.close()


def plot_acc_vs_entropy(acc, entropy_by_prompt, experiment_name, fig_dir):
    common_prompts = set(acc.keys()) & set(entropy_by_prompt.keys())
    acc_vals = np.array([acc[p] for p in common_prompts])
    ent_vals = np.array([entropy_by_prompt[p] for p in common_prompts])
    print(f"{experiment_name} (entropy) - Number of common prompts: {len(common_prompts)}")

    X = ent_vals.reshape(-1, 1)
    y = acc_vals
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    print("Linear Regression Fit (accuracy vs entropy):")
    print("Slope:", reg.coef_[0])
    print("Intercept:", reg.intercept_)
    print("R^2:", r2)
    print("\n")

    x_line = np.linspace(ent_vals.min(), ent_vals.max(), 200).reshape(-1, 1)
    y_line = reg.predict(x_line)

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(ent_vals, acc_vals, alpha=0.6)
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Linear Fit (R^2={r2:.3f})")
    plt.xlabel("Answer entropy (nats, categorical over extracted answers)")
    plt.ylabel("Accuracy")
    pad = 0.05 * max(ent_vals.max() - ent_vals.min(), 1e-6)
    plt.xlim(ent_vals.min() - pad, ent_vals.max() + pad)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f"{experiment_name}_acc_vs_entropy.png"), dpi=300)
    plt.close()


def plot_heatmap(dict1, dict2, model1, model2, fig_dir, metric="accuracy"):
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
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(
        os.path.join(fig_dir, f"{model1}_{model2}_{metric}_heatmap.png"),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


if __name__ == "__main__":
    model_name = "Llama-3.2-1B-Instruct"
    fig_dir = os.path.join("Figs", model_name)

    jsonl_path = "/raid/changl9/eval/base/Llama-3.2-1B-Instruct/0.jsonl"
    base_acc = per_prompt_output_accuracy(jsonl_path)
    base_diversity = per_prompt_output_diversity(jsonl_path)
    base_entropy = per_prompt_output_entropy(jsonl_path)
    bm, bM, bn, bp = overall_accuracy(jsonl_path)
    print(
        f"[Base] overall acc: micro={bm:.4f} macro={bM:.4f} "
        f"(n_rollouts={bn}, n_prompts={bp})"
    )
    base_passk = pass_at_k_means(jsonl_path)
    if base_passk:
        print(f"[Base] pass@k (mean over prompts): {base_passk}")

    jsonl_path = "/raid/changl9/eval/ttrl-verl/ttrl/Llama-3.2-1B-Instruct/0402-125833/global_step_58/0.jsonl"
    ttrl_acc = per_prompt_output_accuracy(jsonl_path)
    ttrl_diversity = per_prompt_output_diversity(jsonl_path)
    ttrl_entropy = per_prompt_output_entropy(jsonl_path)
    tm, tM, tn, tp = overall_accuracy(jsonl_path)
    print(
        f"[TTRL] overall acc: micro={tm:.4f} macro={tM:.4f} "
        f"(n_rollouts={tn}, n_prompts={tp})"
    )
    ttrl_passk = pass_at_k_means(jsonl_path)
    if ttrl_passk:
        print(f"[TTRL] pass@k (mean over prompts): {ttrl_passk}")

    # jsonl_path = "/raid/xinyul2/eval/grpo-naive/Qwen3-4B-Base/0307-173622/global_step_58/0.jsonl"
    # gt_acc = per_prompt_output_accuracy(jsonl_path)
    # gt_diversity = per_prompt_output_diversity(jsonl_path)

    # jsonl_path = "/raid/xinyul2/eval/grpo-intuitor/Qwen3-4B-Base/0307-135909/global_step_58/0.jsonl"
    # intuitor_acc = per_prompt_output_accuracy(jsonl_path)
    # intuitor_diversity = per_prompt_output_diversity(jsonl_path)

    # # check if diversity is a good proxy of accuracy
    plot_acc_vs_div(base_acc, base_diversity, "Base", fig_dir)
    plot_acc_vs_div(ttrl_acc, ttrl_diversity, "TTRL", fig_dir)
    # plot_acc_vs_div(gt_acc, gt_diversity, "Ground Truth", fig_dir)
    # plot_acc_vs_div(intuitor_acc, intuitor_diversity, "Intuitor", fig_dir)

    # # accuracy vs entropy (categorical over extracted answers)
    plot_acc_vs_entropy(base_acc, base_entropy, "Base", fig_dir)
    plot_acc_vs_entropy(ttrl_acc, ttrl_entropy, "TTRL", fig_dir)

    # # check how output accuracy changed after training
    plot_heatmap(base_acc, ttrl_acc, "Base", "TTRL", fig_dir, metric="accuracy")
    # plot_heatmap(base_acc, intuitor_acc, "Base", "Intuitor", fig_dir, metric="accuracy")
    # plot_heatmap(gt_acc, intuitor_acc, "Ground Truth", "Intuitor", fig_dir, metric="accuracy")

    # # check how output diversity changed after training
    plot_heatmap(base_diversity, ttrl_diversity, "Base", "TTRL", fig_dir, metric="diversity")
    # plot_heatmap(base_diversity, intuitor_diversity, "Base", "Intuitor", fig_dir, metric="diversity")
    # plot_heatmap(gt_diversity, intuitor_diversity, "Ground Truth", "Intuitor", fig_dir, metric="diversity")
