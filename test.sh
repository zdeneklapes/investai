function test_all() {
    source venv3.10/bin/activate
    OK_SCRIPTS=("")
    ERROR_SCRIPTS=("")

    # train.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=1 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=1 --test=1 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=1 --test=1 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv out/dataset/stocktadailydataset.csv out/dataset/stockcombineddailydataset.csv --wandb=1 --wandb-sweep=1 --wandb-sweep-count=10 --algorithms ppo a2c td3 sac dqn ddpg --project-verbose="id" --train-verbose=1 --wandb-verbose=1 --total-timesteps=100000 --train=1 --test=1 --env-id=1 --wandb-run-group="sweep-nasfit-1" --baseline-path=out/baseline/baseline.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=2 --test=1 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv --seed=2022 --project-debug
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=2 --test=1 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=2 --train-verbose=1 --wandb-verbose=1 --total-timesteps=2000 --train=3 --test=1 --env-id=1 --wandb-run-group="r1" --baseline-path=out/baseline/baseline.csv --learning-rate=0.0001 --gamma=0.8
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=2 --train-verbose=1 --wandb-verbose=1 -- total-timesteps=2000 --train=2 --test=1 --env-id=1 --wandb-run-group="r1" --baseline-path=out/baseline/baseline.csv --learning-ra
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 -- total-timesteps=1000 --train=1 --test=2 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --algorithms sac --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --total-timesteps=1000 --train=1 --test=1 --env-id=1 --wandb-run-group="r2" --baseline-path=out/baseline/baseline.csv --seed=2022 --project-debug
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --project-verbose="id"

    # report.py

    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py --project-verbose="i" --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv --report-figure
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py --project-verbose=1 --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv --report-figure
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py --project-verbose="i" --baseline-path=out/bas eline/baseline.csv --history-path=out/model/history.csv --report-figure

    # robustness.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/robustness.py --dataset-paths out/dataset/stockfadailyd ataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --project-verbose="id" --train-verbose=1 --wandb-verbose=1 --train=1 - -test=1 --env-id=1 --wandb-run-group="run-robust-1" --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/robustness.py --dataset-paths out/dataset/stockfadailyd ataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --train=1 --test=1 --env-id=1 --wandb-run-group="run-robust-1" --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv --project-debug
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/robustness.py --dataset-paths out/dataset/stockfadailydataset.csv --wandb=1 --wandb-sweep=0 --wandb-sweep-count=1 --project-verbose=1 --train-verbose=1 --wandb-verbose=1 --train=1 --test=1 --env-id=1 --wandb-run-group="run-robust-1" --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv

    # baseline.py
    # TODO

    # stockfadailydataset.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stockfadailydataset.py --project-verbose="i" -dp=out/dataset/stockfadailydataset_1.csv --wandb=1

    # stocktadailydataset.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stocktadailydataset.py --project-verbose="i" -dp=out/dataset/stocktadailydataset_1.csv --wandb=1
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stocktadailydataset.py --project-verbose=1 -dp=out/dataset/stocktadailydataset_1.csv --wandb=1

    # stockcombineddailydataset.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py --project-verbose="id"

    # wandbapi.py
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/wandbapi.py --project-verbose=1 --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv

    [ ${?} -eq 0 ] && OK_SCRIPTS+=("report.py") || ERROR_SCRIPTS+=("report.py")

    echo """Ok Scripts(${#OK_SCRIPTS[*]}): ${OK_SCRIPTS[*]}"""
    echo """Bad Scripts(${#BAD_SCRIPTS[*]}): ${BAD_SCRIPTS[*]}"""
}

test_all
