# INVESTAI - README

---

## AUTHORS

- **Student**: Zdeněk Lapeš <lapes.zdenek@gmail.com>

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

### 3. Run

```shell
# TODO
```

## NOTES

- [MATERIALS](./MATERIALS.md)
