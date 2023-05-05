# INVESTAI - README

---

## AUTHOR

- Zdeněk Lapeš <lapes.zdenek@gmail.com>

## Supervisor

- Milan Češka <ceskam@fit.vut.cz>

## ABOUT


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

In the root directory create `.env` file with the following content and fill in the values, especially `WANDB_API_KEY`:

```shell
# W&B
WANDB_API_KEY=''
WANDB_ENTITY='investai'
WANDB_PROJECT='portfolio-allocation'
WANDB_TAGS='["None"]'
WANDB_JOB_TYPE='train'
WANDB_RUN_GROUP='exp-1'
WANDB_MODE='online'
WANDB_DIR='${PWD}/out/model'


# CUDA - for TensorRT to find CUDA libraries
LD_LIBRARY_PATH='${LD_LIBRARY_PATH}:${HOME}/venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/:${HOME}/venv/lib/python3.10/site-packages/tensorrt/'
```

### 3. Run in Local (from root directory)

```shell
PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/train.py --help

# Examples:
PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/train.py --dataset-path out/dataset/stockfadailydataset.csv --wandb=1 --algorithms ppo --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=1 --test=1 --portfolio-allocation-env=1 --wandb-run-group="exp-run-1" --baseline-path=out/baseline/baseline.csv
```
