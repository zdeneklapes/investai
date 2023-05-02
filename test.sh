function prepare_all_files() {

}

function test_all() {
    source venv3.10/bin/activate
    OK_SCRIPTS=()
    ERROR_SCRIPTS=()

    cmds=(
        # train.py: Run
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py  \
            --dataset-paths out/dataset/stockfadailydataset.csv \
            --algorithms sac \
            --project-verbose=1 \
            --train-verbose=1 \
            --total-timesteps=1000 \
            --train=1 \
            --test=1 \
            --env-id=1 \
            --wandb=1 \
            --wandb-sweep=0 \
            --wandb-sweep-count=1 \
            --wandb-run-group='test-1' \
            --wandb-verbose=1 \
            --baseline-path=out/baseline/baseline.csv"

        # train.py: Sweep
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py \
            --dataset-paths out/dataset/stockfadailydataset.csv out/dataset/stocktadailydataset.csv out/dataset/stockcombineddailydataset.csv \
            --wandb=1 \
            --wandb-sweep=1 \
            --wandb-sweep-count=1 \
            --algorithms ppo a2c td3 sac dqn ddpg \
            --project-verbose='id' \
            --train-verbose=1 \
            --wandb-verbose=1 \
            --total-timesteps=1000 \
            --train=1 \
            --test=1 \
            --env-id=1 \
            --wandb-run-group='test-sweep-1' \
            --baseline-path=out/baseline/baseline.csv"

        # report.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py \
            --project-verbose='i' \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv \
            --report-figure"

        # robustness.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/robustness.py \
            --dataset-paths out/dataset/stockfadailydataset.csv \
            --wandb=1 \
            --wandb-sweep=0 \
            --wandb-sweep-count=1 \
            --project-verbose='id' \
            --train-verbose=1 \
            --wandb-verbose=1 \
            --train=1 \
            --test=1 \
            --env-id=1 \
            --wandb-run-group='run-robust-1' \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv"

        # baseline.py
        "PYTHONPATH=$PWD/investai python3 investai/extra/math/finance/shared/baseline.py \
            --project-verbose='i' \
            --dataset-paths out/dataset/stockfadailydataset.csv \
            --baseline-path=out/baseline/baseline.csv"

        # stockfadailydataset.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stockfadailydataset.py \
        --project-verbose='i' \
        --dataset-paths=out/dataset/stockfadailydataset.csv \
        --wandb=1"

        # stocktadailydataset.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stocktadailydataset.py \
            --project-verbose='i' \
            --dataset-paths=out/dataset/stocktadailydataset.csv \
            --wandb=1"
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stocktadailydataset.py --project-verbose='i' -dp=out/dataset/stocktadailydataset_1.csv --wandb=1"

        # stockcombineddailydataset.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/train.py \
            --project-verbose='id'"

        # wandbapi.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/wandbapi.py \
            --project-verbose=1 \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv"
    )

    #    for cmd in "${cmds[@]}"; do
    for cmd in 1 2 3; do
        echo "Running: ${cmd}"
        eval "${cmd}"
        if [ $? -eq 0 ]; then OK_SCRIPTS+=("${cmd}"); else ERROR_SCRIPTS+=("${cmd}"); fi
    done

    echo "Ok Scripts (${#OK_SCRIPTS[*]})"
    echo "Error Scripts (${#ERROR_SCRIPTS[*]}):"
    for error_script in "${ERROR_SCRIPTS[@]}"; do echo "${error_script}"; done
}

test_all
