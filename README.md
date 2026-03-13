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
python -m examples.data_preprocess.math_dataset_ttt --local_save_dir ~/data/math --data_source DigitalLearningGmbH/MATH-lighteval
```
To prepare MATH-500 for evaluation, run
```bash
python -m examples.data_preprocess.math_dataset_ttt --local_save_dir ~/data/MATH-500 --data_source HuggingFaceH4/MATH-500
```
The `parquet` files will be generated under `~/data/`.

## Datasets

To train a model, run:
```bash
bash examples/test_time_training/run.sh
```

## Evaluate

To evaluate a trained model, first merge the verl checkpoints into a huggingface model:
```bash
bash examples/test_time_training/merge.sh
```
Then run
```bash
bash examples/test_time_training/evaluate.sh
```