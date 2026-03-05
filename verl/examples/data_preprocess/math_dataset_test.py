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
"""
Preprocess the MATH test datasets to parquet format
"""

import os
import datasets


def make_map_fn(split, source=None):
    def process_fn(example, idx):
        if source is None:
            data_source = example.pop("source")
        else:
            data_source = source

        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question = example.pop("prompt")
        question = question + " " + instruction_following
        solution = example.pop("answer")

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
        return data

    return process_fn


if __name__ == "__main__":
    # all math benchmarks
    data_sources = ['AIME24', 'AIME25', 'AMC', 'MATH-500']

    for data_source in data_sources:
        print(f"Processing dataset: {data_source}")

        test_file = os.path.join("data", data_source, 'test.json')
        if not os.path.exists(test_file):
            print(f"Warning: Test file {test_file} does not exist, skipping")
            continue

        try:
            test_dataset = datasets.load_dataset("json", data_files=test_file, split='train')
            test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)
            
            local_dir = os.path.expanduser("~/data")
            os.makedirs(os.path.join(local_dir, data_source), exist_ok=True)
            test_dataset.to_parquet(os.path.join(local_dir, data_source, "test.parquet"))
            
            print(f"Successfully processed dataset: {data_source}")
        
        except Exception as e:
            print(f"Error processing dataset {data_source}: {e}")
            continue