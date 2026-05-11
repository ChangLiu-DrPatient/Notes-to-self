# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter, defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.auto_extract import extract_answer, get_source_family
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("ttrl")
class TTRLRewardManager(AbstractRewardManager):
    """TTRL Reward Manager.

    Groups responses by prompt, finds the majority answer via voting,
    then scores each response against that majority answer as a pseudo-label
    rather than against the ground truth.

    Args:
        tokenizer: The tokenizer used to decode token IDs into text.
        num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
        compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
        reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
            "data_source".
        n_votes_per_prompt: The number of votes per prompt for rollout.
        n_samples_per_prompt: The number of samples per prompt for advantage estimation.
        eval_n_samples: The number of validation samples per prompt when aggregating diagnostics.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        n_votes_per_prompt: int = 8,
        n_samples_per_prompt: int = 8,
        eval_n_samples: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.eval_n_samples = eval_n_samples

        if self.n_votes_per_prompt < self.n_samples_per_prompt:
            raise ValueError(
                "For TTRL settings, "
                f"n_votes_per_prompt {self.n_votes_per_prompt} should be greater than or equal to "
                f"n_samples_per_prompt {self.n_samples_per_prompt}"
            )

    def _decode_item(self, data_item) -> tuple[str, str, int]:
        """Decode a single data item, returning prompt str, response str, valid response length."""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        return prompt_str, response_str, int(valid_response_length)

    def _score(self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None) -> float:
        """Thin wrapper around compute_score that always returns a float."""
        score = self.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return float(score["score"] if isinstance(score, dict) else score)

    def _majority_vote_rewards(
        self,
        data_source: str,
        outputs: list[str],
        labels: list[str],
        extra_infos: list[dict | None],
    ) -> tuple[list[float], dict]:
        """
        1. Extract answer from each output using data_source-specific extractor
        2. Find majority answer by Counter
        3. Score each output against majority answer (pseudo ground truth)
        4. Compute TTRL diagnostic metrics against true ground truth
        """
        n = len(outputs)
        gt = labels[0]  # all labels identical within a prompt group

        # step 1: extract answers
        extracted = [
            extract_answer(data_source, output, extra)
            for output, extra in zip(outputs, extra_infos)
        ]
        valid_answers = [a for a in extracted if a is not None]

        # edge case: no valid answers extracted from any response
        if not valid_answers:
            rewards = [-1.0] * n
            ttrl_metrics = {
                "label_accuracy": 0.0,
                "majority_ratio": 0.0,
                "ground_truth_reward": 0.0,
                "majority_voting_reward": -1.0,
                f"pass@{n}": 0.0,
            }
            return rewards, ttrl_metrics

        # step 2: majority vote
        counter = Counter(valid_answers)
        estimated_label, majority_count = counter.most_common(1)[0]

        # step 3: reward = agreement with majority pseudo-label
        rewards = [
            self._score(data_source, output, estimated_label, extra)
            for output, extra in zip(outputs, extra_infos)
        ]

        # step 4: diagnostic metrics against ground truth (not used as reward signal)
        true_rewards = [
            self._score(data_source, output, gt, extra)
            for output, extra in zip(outputs, extra_infos)
        ]

        # check if majority label itself is correct
        # wrap in \boxed{} so math scorers can extract it cleanly
        label_accuracy = self._score(
            data_source, f"\\boxed{{{estimated_label}}}", gt, None
        )

        ttrl_metrics = {
            "label_accuracy": label_accuracy,
            "majority_ratio": majority_count / n,
            "ground_truth_reward": sum(true_rewards) / n,
            "majority_voting_reward": sum(rewards) / n,
            f"pass@{n}": 1.0 if sum(true_rewards) >= 1 else 0.0,
        }

        return rewards, ttrl_metrics

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # pass through rm scores directly if present
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_extra_info = defaultdict(list)
        # Validation batches are marked by the trainer; everything else uses train-time TTRL rewards.
        is_validate = bool(data.meta_info.get("validate", False))

        if not is_validate:
            # Training path: group rollouts by prompt, majority-vote a pseudo-label, then reward each vote.
            assert len(data) % self.n_votes_per_prompt == 0, (
                f"Data length {len(data)} must be divisible by "
                f"n_votes_per_prompt {self.n_votes_per_prompt}"
            )

            prompt_num = len(data) // self.n_votes_per_prompt

            # Reward tensor and extras only cover the kept training samples.
            reward_tensor = torch.zeros(
                (prompt_num * self.n_samples_per_prompt, data.batch["responses"].shape[-1]),
                dtype=torch.float32,
                device=data.batch["responses"].device,
            )

            already_print_data_sources: dict[str, int] = {}

            for prompt_i in range(prompt_num):
                # Collect all candidate responses for one prompt before running majority voting.
                group_outputs: list[str] = []
                group_labels: list[str] = []
                group_extra_infos: list[dict | None] = []
                group_valid_lengths: list[int] = []
                data_source: str | None = None
                data_source_family: str | None = None
                first_prompt_str: str = ""

                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    prompt_str, response_str, valid_response_length = self._decode_item(data_item)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    extra_info = data_item.non_tensor_batch.get("extra_info", {})
                    current_data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    current_data_source_family = get_source_family(current_data_source)

                    if data_source is None:
                        data_source = current_data_source
                        data_source_family = current_data_source_family
                        first_prompt_str = prompt_str
                    elif current_data_source_family != data_source_family:
                        raise ValueError(
                            "All train-time TTRL samples in a prompt group must share the same source family. "
                            f"got first_data_source={data_source!r} ({data_source_family!r}) and "
                            f"current_data_source={current_data_source!r} ({current_data_source_family!r})"
                        )

                    group_outputs.append(response_str)
                    group_labels.append(ground_truth)
                    group_extra_infos.append(extra_info)
                    group_valid_lengths.append(valid_response_length)

                rewards, ttrl_metrics = self._majority_vote_rewards(
                    data_source, group_outputs, group_labels, group_extra_infos
                )

                # Broadcast prompt-level diagnostics onto each kept sample so reward_extra_info keeps the
                # same row-aligned shape expected from other reward managers.
                for k, v in ttrl_metrics.items():
                    reward_extra_info[k].extend([v] * self.n_samples_per_prompt)

                for i in range(self.n_samples_per_prompt):
                    train_idx = prompt_i * self.n_samples_per_prompt + i
                    reward_tensor[train_idx, group_valid_lengths[i] - 1] = rewards[i]
                    reward_extra_info["acc"].append(rewards[i])

                for i in range(self.n_votes_per_prompt):
                    # restore the earlier per-sample debug printing behavior
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0
                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", first_prompt_str)
                        print("[response]", group_outputs[i])
                        print("[ground_truth]", group_labels[i])
                        print("[majority_reward]", rewards[i])
                        print("[ttrl_metrics]", ttrl_metrics)
        else:
            # Validation path: score each sample against ground truth, then compute grouped TTRL diagnostics.
            assert len(data) % self.eval_n_samples == 0, (
                f"Data length {len(data)} must be divisible by "
                f"eval_n_samples {self.eval_n_samples}"
            )

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print_data_sources: dict[str, int] = {}
            eval_samples: list[dict[str, Any]] = []
            # Group by data source so mixed validation batches can still be scored safely.
            data_source_groups: dict[str, dict[str, list[Any]]] = {}

            for sample_idx in range(len(data)):
                data_item = data[sample_idx]
                prompt_str, response_str, valid_response_length = self._decode_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                eval_samples.append(
                    {
                        "sample_idx": sample_idx,
                        "prompt": prompt_str,
                        "response": response_str,
                        "ground_truth": ground_truth,
                        "data_source": data_source,
                        "extra_info": extra_info,
                        "valid_response_length": valid_response_length,
                    }
                )

                if data_source not in data_source_groups:
                    data_source_groups[data_source] = {
                        "indices": [],
                        "outputs": [],
                        "labels": [],
                        "extra_infos": [],
                        "valid_response_lengths": [],
                    }
                data_source_groups[data_source]["indices"].append(sample_idx)
                data_source_groups[data_source]["outputs"].append(response_str)
                data_source_groups[data_source]["labels"].append(ground_truth)
                data_source_groups[data_source]["extra_infos"].append(extra_info)
                data_source_groups[data_source]["valid_response_lengths"].append(valid_response_length)

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)

            eval_rewards: list[float] = [0.0] * len(data)
            eval_preds: list[str | None] = [None] * len(data)
            for data_source, group in data_source_groups.items():
                # Backfill ground-truth rewards to the original sample order after grouped scoring.
                for idx_in_group, sample_idx in enumerate(group["indices"]):
                    response_str = group["outputs"][idx_in_group]
                    ground_truth = group["labels"][idx_in_group]
                    extra_info = group["extra_infos"][idx_in_group]
                    valid_response_length = group["valid_response_lengths"][idx_in_group]
                    reward = self._score(data_source, response_str, ground_truth, extra_info)

                    reward_tensor[sample_idx, valid_response_length - 1] = reward
                    eval_rewards[sample_idx] = reward
                    eval_preds[sample_idx] = extract_answer(data_source, response_str, extra_info)

            reward_extra_info["acc"].extend(eval_rewards)
            reward_extra_info["pred"].extend(eval_preds)

            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                # Broadcast prompt-level eval diagnostics across the prompt's repeated samples so the generic
                # validation pipeline can treat them like other row-aligned reward_extra_info fields.
                group_samples = eval_samples[
                    prompt_i * self.eval_n_samples : (prompt_i + 1) * self.eval_n_samples
                ]
                data_source = group_samples[0]["data_source"]
                data_source_family = get_source_family(data_source)

                for sample in group_samples[1:]:
                    sample_data_source = sample["data_source"]
                    sample_data_source_family = get_source_family(sample_data_source)
                    if sample_data_source_family != data_source_family:
                        raise ValueError(
                            "All eval samples in a prompt group must share the same source family. "
                            f"got first_data_source={data_source!r} ({data_source_family!r}) and "
                            f"current_data_source={sample_data_source!r} ({sample_data_source_family!r})"
                        )

                _, ttrl_metrics = self._majority_vote_rewards(
                    data_source=data_source,
                    outputs=[sample["response"] for sample in group_samples],
                    labels=[sample["ground_truth"] for sample in group_samples],
                    extra_infos=[sample["extra_info"] for sample in group_samples],
                )

                for k, v in ttrl_metrics.items():
                    reward_extra_info[k].extend([v] * self.eval_n_samples)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor