OK_SCRIPTS=()
ERROR_SCRIPTS=()

function prepare_all_files() {
    rm -rf out
    mkdir -p out/baseline out/dataset out/model
    source venv/bin/activate
    wandb artifact get investai/portfolio-allocation/stockfadailydataset:latest --root out/dataset
    wandb artifact get investai/portfolio-allocation/stocktadailydataset:latest --root out/dataset
    wandb artifact get investai/portfolio-allocation/stockcombineddailydataset:latest --root out/dataset

    # Baseline
    wandb artifact get investai/portfolio-allocation/baseline:latest --root out/baseline

    # History
    wandb artifact get investai/portfolio-allocation/history:latest --root out/model
}

function test_datasets() {
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

        # stockcombineddailydataset.py
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/dataset/stockcombinedailydataset.py \
            --project-verbose='i' \
            --dataset-paths \
                out/dataset/stockfadailydataset.csv \
                out/dataset/stocktadailydataset.csv \
                out/dataset/stockcombineddailydataset.csv \
            --wandb=1"
    )
    for cmd in "${cmds[@]}"; do
        echo "Running: ${cmd}"
        eval "${cmd}"
        if [ $? -eq 0 ]; then OK_SCRIPTS+=("${cmd}"); else ERROR_SCRIPTS+=("${cmd}"); fi
    done
}

function test_other() {
    source venv/bin/activate
    cmds=(
        # baseline.py --  baseline.csv
        "PYTHONPATH=$PWD/investai python3 investai/extra/math/finance/shared/baseline.py \
            --project-verbose='i' \
            --dataset-paths out/dataset/stockfadailydataset.csv \
            --baseline-path=out/baseline/baseline.csv"

        # wandbapi.py -- history.csv
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

        # report.py -- figure/
        "PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py \
            --project-verbose='i' \
            --baseline-path=out/baseline/baseline.csv \
            --history-path=out/model/history.csv \
            --report-figure"
    )

    for cmd in "${cmds[@]}"; do
        echo "Running: ${cmd}"
        eval "${cmd}"
        if [ $? -eq 0 ]; then OK_SCRIPTS+=("${cmd}"); else ERROR_SCRIPTS+=("${cmd}"); fi
    done

}

function usage() {
    echo "USAGE:
'--prepare-files') prepare_all_files ;; # Prepare all files, datasets, baseline, history (download from wandb)
'--test-dataset') test_datasets ;;     # Test datasets creation
'--test-other') test_other ;;          # Test other scripts
"
}

[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
    '--prepare-files') prepare_all_files ;; # Prepare all files, datasets, baseline, history (download from wandb)
    '--test-dataset') test_datasets ;;      # Test datasets creation
    '--test-other') test_other ;;           # Test other scripts
    esac
    shift
done

#
echo "Ok Scripts (${#OK_SCRIPTS[*]})"
echo "Error Scripts (${#ERROR_SCRIPTS[*]}):"
for error_script in "${ERROR_SCRIPTS[@]}"; do echo "${error_script}"; done
