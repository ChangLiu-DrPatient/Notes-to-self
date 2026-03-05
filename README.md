# Test-time Training

## Env Setup

```bash
git clone https://github.com/moment-timeseries-foundation-model/Test-Time-Training.git

cd Test-Time-Training/verl

conda create -n verl python==3.12
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install -U "numpy==2.2.0"  # resolve conflicts with Numba
pip install --no-deps -e .
```

## Datasets

To prepare the training dataset, run:
```bash
python -m examples.data_preprocess.math_dataset
```
The `json` files for AIME24, AIME25, AMC, and MATH-500 are under `verl/data` (copied from EVOL-RL repo). To prepare the test datasets, run:
```bash
python examples/data_preprocess/math_dataset_test.py
```
The `parquet` files will be generated under `~/data/`.
