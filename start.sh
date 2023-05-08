#!/bin/bash
#set -x # log

RM="rm -rfd"
RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

AUTHOR='Zdenek Lapes'
EMAIL='lapes.zdenek@gmail.com'

LOGIN="xlapes02"

################################################################################
# Functions
################################################################################
function error_exit() {
    printf "${RED}ERROR: $1${NC}\n"
    usage
    exit 1
}

function project_clean() {
    ${RM} *.zip

    # Folders
    for folder in "venv" "__pycache__" "migrations"; do
        find . -type d -iname "${folder}" | xargs "${RM}"
    done

    # Files
    for file in ".DS_Store"; do
        find . -type f -iname "${file}" | xargs "${RM}"
    done
}

function tags() {
    ctags -R --fields=+l \
        --exclude=.git \
        --exclude=.idea \
        --exclude=node_modules \
        --exclude=tests* \
        --exclude=venv* .
    cscope -Rb
}

function upload_code() {
    rsync -avPz \
        --exclude-from=.rsync_ignore_code \
        ./src ./requirements.txt \
        xlapes02@sc-gpu1.fit.vutbr.cz:/home/xlapes02/ai-investing
}

function usage() {
    echo "USAGE:
    '-c' | '--clean') clean ;;
    '-cd' | '--clean-docker') clean_docker ;;
    '-id' | '--install-docker') install_docker ;;
    '-idd' | '--install-docker-deploy') install_docker_deploy ;;
    '-dsip' | '--install-docker-deploy') docker_show_ipaddress ;;
        #
    '--create-samples-env') create_env_samples ;;
    '-sc' | '--sync-code') upload_code ;;
        #
    '-t' | '--tags') tags ;;
    '-h' | '--help') usage ;;"
}

function project_pack() {
    zip -r "${LOGIN}.zip" \
        investai \
        Dockerfile \
        requirements.txt \
        start.sh \
        test.sh \
        README.md \
        .pre-commit-config.yaml \
        .editorconfig \
        pyproject.toml \
        .git/.gitkeep \
        \
        -x \
        **__pycache__** \
        **pytest_cache** \
        **.DS_Store** \
        **.ruff_cache** \
        **.vscode** \
        **tags**
    mv "${LOGIN}.zip" "${HOME}/Downloads/"
    unzip -d "${HOME}/Downloads/${LOGIN}" "${HOME}/Downloads/${LOGIN}.zip"
    open "${HOME}/Downloads/${LOGIN}"
}

function project_pack_all() {
    zip -r "${LOGIN}.zip" \
        out/dataset \
        out/baseline \
        out/model/history.csv \
        investai \
        Dockerfile \
        requirements.txt \
        start.sh \
        test.sh \
        README.md \
        .pre-commit-config.yaml \
        .editorconfig \
        pyproject.toml \
        .git/.gitkeep \
        \
        -x \
        **__pycache__** \
        **pytest_cache** \
        **.DS_Store** \
        **.ruff_cache** \
        **.vscode** \
        **tags**
    mv "${LOGIN}.zip" "${HOME}/Downloads/"
    unzip -d "${HOME}/Downloads/${LOGIN}" "${HOME}/Downloads/${LOGIN}.zip"
    open "${HOME}/Downloads/${LOGIN}"
}

function create_env_samples() {
    cd env || error_exit "cd"

    # Clean all samples
    find . -type f -iname "sample*" | xargs "${RM}"

    # Create new samples
    for f in $(find . -type f -iname ".env*" | cut -d/ -f2); do
        cat "${f}" | cut -d "=" -f1 >"sample${f}"
    done

    cd .. || error_exit "cd"
}

################################################################################
# DOCKER
################################################################################
DOCKER_CONTAINER_VERSION="0.5"
DOCKER_NAME="python/testing_env"
DOCKER_CONTAINER_NAME="python_testing_env"

function docker_build() {
    docker build -t "${DOCKER_NAME}:${DOCKER_CONTAINER_VERSION}" -f Dockerfile .
}

function docker_run() {
    docker run -itd --cap-add sys_ptrace -p 127.0.0.1:2222:22 --name ${DOCKER_CONTAINER_NAME} -v "$(pwd)":/home/user/project "${DOCKER_NAME}:${DOCKER_CONTAINER_VERSION}"
}

function docker_start() {
    docker start "${DOCKER_CONTAINER_NAME}"
}

function docker_stop() {
    docker stop "${DOCKER_CONTAINER_NAME}"
}

function docker_stop_remove_container() {
    docker stop $(docker ps -q) && docker rm clion_remote_env
}

function docker_show_ipaddress() {
    for docker_container in $(docker ps -aq); do
        CMD1="$(docker ps -a | grep "$docker_container" | grep --invert-match "Exited\|Created" | awk '{print $2}'): "
        if [ "$CMD1" != ": " ]; then
            printf "$CMD1"
            printf "$(docker inspect ${docker_container} | grep "IPAddress" | tail -n 1)\n"
        fi
    done
}

function docker_clean() {
    docker stop ${DOCKER_CONTAINER_NAME}
    docker rm ${DOCKER_CONTAINER_NAME}
    docker rmi ${DOCKER_NAME}:${DOCKER_CONTAINER_VERSION}
}
function docker_clean_all() {
    docker stop $(docker ps -aq)
    docker system prune -a -f
    docker volume prune -f
}

function requirement_for_workflow() {
    # Because "tvdatafeed" is not available on PyPi for Python 3.10
    #    source venv/bin/activate && pip-chill >requirements.txt
    file_name="requirements_for_workflows.txt"
    cat requirements.txt | grep --invert-match "tvdatafeed\|finrl-meta\|pyfolio" >${file_name}
    git add ${file_name}
}

function run_test() {
    #    source venv/bin/activate &&
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/wandb_train.py --dataset-path=out/datasets/stockfadailydataset.csv --wandb-project="investai_exp_1" --wandb-job-type="train" --wandb=1 --wandb-sweep=0 --wandb-sweep-count=2 --algorithms ppo --project-verbose=1 --total-timesteps=1000
}

function run_sweep() {
    #    source venv/bin/activate &&
    PYTHONPATH=$PWD/investai python3 investai/run/portfolio_allocation/train/wandb_train.py --dataset-path=out/datasets/stockfadailydataset.csv --wandb-project="investai_sweep_1" --wandb-job-type="train" --wandb=0 --wandb-sweep=1 --wandb-sweep-count=100 --algorithms ppo a2c sac td3 dqn ddpg --project-verbose=1 --total-timesteps=400000
}

function copy_figures_to_ibt_thesis() {
    cp -r out/figure/ ../ibt/thesis/image/figure/
}

function install() {
    mkdir -p out/baseline out/dataset out/model
    VENV_NAME="venv"
    python3 -m venv ${VENV_NAME}
    source ${VENV_NAME}/bin/activate
    for i in $(cat requirements.txt | cut -d '=' -f1 | grep --invert-match "^#"); do pip3 install $i; done
}

function foo() {
    docker build -t investai -f Dockerfile .
}

##### PARSE CLI-ARGS
[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
    # "-p.*" prefix all project commands
    '-pc' | '--project-clean') project_clean ;;
    '-pp' | '--project-pack') project_pack ;;
    '-pp' | '--project-pack-all') project_pack_all ;;
    '-prfw' | '--project-requirements-for-workflow') requirement_for_workflow ;;
    '-pf' | '--project-figures') copy_figures_to_ibt_thesis ;;
    '-pi' | '--project-install') install ;;
        # "-d.*" prefix all docker commands
    '-dr' | '--docker-run') docker_run ;;
    '--docker-start') docker_start ;;
    '--docker-stop') docker_stop ;;
    '-db' | '--docker-build') docker_build ;;
    '-dc' | '--docker-clean') docker_clean ;;
    '-dca' | '--docker-clean-all') docker_clean_all ;;
    '-dsrc' | '--docker-stop-remove-container') docker_stop_remove_container ;;
        #
    '-rt' | '--run-test') run_test ;;
    '-rs' | '--run-sweep') run_sweep ;;
    esac
    shift
done
