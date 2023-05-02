function prepare_all_files() {
    rm -rf out
    mkdir -p out/baseline out/dataset out/model
    wandb artifact get investai/portfolio-allocation/stockfadailydataset:latest --root out/dataset
    wandb artifact get investai/portfolio-allocation/stocktadailydataset:latest --root out/dataset
    wandb artifact get investai/portfolio-allocation/stockcombineddailydataset:latest --root out/dataset

    # Baseline
    wandb artifact get investai/portfolio-allocation/baseline:latest --root out/baseline

    # History
    wandb artifact get investai/portfolio-allocation/history:latest --root out/model
}

function test_all() {
    source venv3.10/bin/activate
    OK_SCRIPTS=()
    ERROR_SCRIPTS=()

    cmds=(
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

        # baseline.py
        "PYTHONPATH=$PWD/investai python3 investai/extra/math/finance/shared/baseline.py \
            --project-verbose='i' \
            --dataset-paths out/dataset/stockfadailydataset.csv \
            --baseline-path=out/baseline/baseline.csv"

        # wandbapi.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/wandbapi.py \
            --project-verbose=1 \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv"

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
            --algorithms ppo a2c \
            --project-verbose='id' \
            --train-verbose=1 \
            --total-timesteps=1000 \
            --train=1 \
            --test=1 \
            --env-id=1 \
            --wandb=1 \
            --wandb-sweep=1 \
            --wandb-sweep-count=1 \
            --wandb-run-group='test-sweep-1' \
            --wandb-verbose=1 \
            --baseline-path=out/baseline/baseline.csv"

        # report.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py \
            --project-verbose='i' \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv \
            --report-figure"

        # robustness.py
        #        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/robustness.py \
        #            --dataset-paths out/dataset/stockfadailydataset.csv \
        #            --wandb=1 \
        #            --wandb-sweep=0 \
        #            --wandb-sweep-count=1 \
        #            --project-verbose='id' \
        #            --train-verbose=1 \
        #            --wandb-verbose=1 \
        #            --train=1 \
        #            --test=1 \
        #            --env-id=1 \
        #            --wandb-run-group='run-robust-1' \
        #            --baseline-path=out/baseline/baseline.csv \
        #            --history-path=out/model/history.csv"

    )

    #    for cmd in "${cmds[@]}"; do
    for cmd in "${cmds[@]}"; do
        echo "Running: ${cmd}"
        eval "${cmd}"
        if [ $? -eq 0 ]; then OK_SCRIPTS+=("${cmd}"); else ERROR_SCRIPTS+=("${cmd}"); fi
    done

    echo "Ok Scripts (${#OK_SCRIPTS[*]})"
    echo "Error Scripts (${#ERROR_SCRIPTS[*]}):"
    for error_script in "${ERROR_SCRIPTS[@]}"; do echo "${error_script}"; done
}

[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
    '--prepare-files') prepare_all_files ;;
    '--test') test_all ;;
    esac
    shift
done
