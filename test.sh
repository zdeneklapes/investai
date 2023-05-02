function test_all() {
    source venv3.10/bin/activate
    OK_SCRIPTS=("")
    ERROR_SCRIPTS=("")

    # train.py

    # report.py
    #    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/thesis/report.py --project-verbose="i" --baseline-path=out/baseline/baseline.csv --history-path=out/model/history.csv --report-figure
    echo "TODO"
    [ ${?} -eq 0 ] && OK_SCRIPTS+=("report.py") || ERROR_SCRIPTS+=("report.py")

    echo """Ok Scripts(${#OK_SCRIPTS[*]}): ${OK_SCRIPTS[*]}"""
    echo """Bad Scripts(${#BAD_SCRIPTS[*]}): ${BAD_SCRIPTS[*]}"""
    # robustness.py

    # baseline.py
}

test_all
