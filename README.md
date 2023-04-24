# INVESTAI - README

---

## AUTHOR

- Zdeněk Lapeš <lapes.zdenek@gmail.com>

## DEPENDENCIES

- swig
- ta-lib

## CONTRIBUTING

### Before pull request

```bash
# Pre-commit
pre-commit install              # Run all before commit)
pre-commit install -t pre-push  # Run all before push)
pre-commit run --all-files      # Run all checks manually

# Run tests
pytest
```

## INSTALLATION

### 1. Install dependencies

```shell
# venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip install --upgrade --no-cache-dir git+https://github.com/StreamAlpha/tvdatafeed.git
```

### 2. Create `.env` file

In the root directory create `.env` file with the following content:

```shell
# Data Providers
EOD_HISTORICAL_DATA_API=''
NASDAQ_HISTORICAL_DATA_API=''
ALPHA_VANTAGE_API=''
FINANCIAL_MODELING_PREP_API=''

# W&B
WANDB_API_KEY=''
WANDB_ENTITY=''
WANDB_PROJECT=''
WANDB_TAGS=''
WANDB_JOB_TYPE=''
WANDB_RUN_GROUP=''
WANDB_MODE=''
WANDB_DIR=''

# CUDA - for TensorRT to find CUDA libraries
LD_LIBRARY_PATH='${LD_LIBRARY_PATH}:${HOME}/venv3.10/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/:${HOME}/venv3.10/lib/python3.10/site-packages/tensorrt/'
```

### 3. Run in Local (from root directory)

```shell
PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/train.py --help

# Examples:
PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/train.py --dataset-path out/dataset/stockfadailydataset.csv --wandb=1 --algorithms ppo --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=1 --test=1 --portfolio-allocation-env=1 --wandb-run-group="exp-run-1" --baseline-path=out/baseline/baseline.csv
```

## NOTES

- [MATERIALS](./MATERIALS.md)
