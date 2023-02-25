#!/bin/bash
#set -x # log

RM="rm -rfd"
RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

AUTHOR='Zdenek Lapes'
EMAIL='lapes.zdenek@gmail.com'

PROJECT_NAME='pawnshop'

################################################################################
# Functions
################################################################################
function error_exit() {
    printf "${RED}ERROR: $1${NC}\n"
    usage
    exit 1
}

function clean() {
    ${RM} *.zip

    # Folders
    for folder in "venv" "venv3.10" "__pycache__" "migrations"; do
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

function pack() {
    zip -r thesis.zip \
        thesis \
        -x "*out*" \
        -x "*others*"
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
    file_name="requirements_for_workflows.txt"
    cat requirements.txt | grep --invert-match "tvdatafeed" >${file_name}
    git add ${file_name}
}

##### PARSE CLI-ARGS
[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
    # "-p.*" prefix all project commands
    '-pc' | '--project-clean') clean ;;
    '-prfw' | '--project-requirements-for-workflow') requirement_for_workflow ;;
        # "-d.*" prefix all docker commands
    '-dr' | '--docker-run') docker_run ;;
    '--docker-start') docker_start ;;
    '--docker-stop') docker_stop ;;
    '-db' | '--docker-build') docker_build ;;
    '-dc' | '--docker-clean') docker_clean ;;
    '-dca' | '--docker-clean-all') docker_clean_all ;;
    '-dsrc' | '--docker-stop-remove-container') docker_stop_remove_container ;;

        #    '-cd' | '--clean-docker') clean_docker ;;
        #    '-id' | '--install-docker') install_docker_compose ;;
        #    '-idd' | '--install-docker-deploy') install_docker_deploy ;;
        #    '-dsip' | '--install-docker-deploy') docker_show_ipaddress ;;
        #        #
        #    '--create-samples-env') create_env_samples ;;
        #    '-sc' | '--sync-code') upload_code ;;
        #        #
        #    '-t' | '--tags') tags ;;
        #    '-h' | '--help') usage ;;
        #    '-p' | '--pack') pack ;;
    esac
    shift
done
