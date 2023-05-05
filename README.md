# INVESTAI - Reinforcement Learning for Automated Stock Portfolio Allocation

# README:

---

## AUTHOR

- Zdeněk Lapeš <lapes.zdenek@gmail.com>

## Supervisor

- Milan Češka <ceskam@fit.vut.cz>

## ABOUT

#### This project implements:

- training and testing pipeline of reinforcement learning algorithms for portfolio allocation with connection to W&B
- dataset creation from stock data
- baseline creation from stock data
- creating report for thesis

#### Folder structure `investai/`:

- **investai**: Whole thesis code is here
- **investai/shared**: Shared code throughout the project
- **investai/shared/tests**: Tests for utils.py
- **investai/raw_data**: Downloading and processing of raw data
- **investai/extra/math/finance/shared**: Baseline creation
- **investai/extra/math/finance/ticker**: Downloading and processing of raw data
- **investai/run/portfolio_allocation/thesis**: Training and testing scripts
- **investai/run/portfolio_allocation/thesis/dataset**: Datasets creation scripts
- **investai/run/portfolio_allocation/envs**: Portfolio allocation environments
- **investai/run/shared**: Shared code throughout only for scripts in run folder
- **investai/run/shared/sb3**: Sweep configuration for Stable Baselines3 and algorithms of Stable Baselines3
- **investai/run/shared/dataset**: Shared code for dataset creation
- **investai/run/shared/callback**: Callbacks for training
-

#### Folder structure `out/`:

Out folder is created when whatever script is run. It contains:

- **out**: Output folder
- **out/baseline**: Baseline csv files
- **out/dataset**: Dataset csv files
- **out/figure**: Figure png files and latex files for thesis
- **out/model**: Model files, WandB files, TensorBoard files and history

## INSTALLATION

### 1. Install dependencies

```shell
# venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
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

### 3. Run Program

#### Run Tests

```shell
./test.sh --prepare-files # Remove out folder, and download datasets from W&B (You must be logged in/API key set)
./test.sh --test # Run all scripts and tests if any fail
```

#### Print help

```shell
PYTHONPATH=$PWD/investai python3 \
    investai/run/portfolio_allocation/thesis/train.py \
        --help
```

#### Single Run (train/test)

```
PYTHONPATH=$PWD/investai python3 \
    investai/run/portfolio_allocation/thesis/train.py \
    --dataset-paths out/dataset/stockfadailydataset.csv \
    --algorithms ppo \
    --project-verbose='i' \
    --train-verbose=1 \
    --total-timesteps=1000 \
    --train=1 \
    --test=1 \
    --env-id=1 \
    --wandb=1 \
    --wandb-run-group="exp-run-1" \
    --wandb-verbose=1 \
    --baseline-path=out/baseline/baseline.csv
```

#### Sweep Run: 3 runs with random hyperparameters over 2 datasets and 5 algorithms (train/test)

```shell
PYTHONPATH=$PWD/investai python3 \
  investai/run/portfolio_allocation/thesis/train.py \
  --dataset-paths \
      out/dataset/stockfadailydataset.csv \
      out/dataset/stockcombineddailydataset.csv \
  --algorithms \
      ppo \
      a2c \
      td3 \
      ddpg \
      sac \
  --project-verbose='i' \
  --train-verbose=1 \
  --total-timesteps=1000 \
  --train=1 \
  --test=1 \
  --env-id=1 \
  --wandb=1 \
  --wandb-sweep=1 \
  --wandb-sweep-count=3 \
  --wandb-verbose=1 \
  --wandb-run-group="exp-sweep-1" \
  --baseline-path=out/baseline/baseline.csv
```
