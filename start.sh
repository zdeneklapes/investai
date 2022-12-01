#!/bin/bash
#set -x # log

RM="rm -rfd"
RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

AUTHOR='Zdenek Lapes'
EMAIL='lapes.zdenek@gmail.com'

PROJECT_NAME='pawnshop'

##### FUNCTIONS
function error_exit() {
    printf "${RED}ERROR: $1${NC}\n"
    usage
    exit 1
}

function clean() {
    ${RM} *.zip

    # Folders
    for folder in "venv" "__pycache__" "migrations"; do
        find . -type d -iname "${folder}" | xargs "${RM}"
    done

    # Files
    for file in ".DS_Store" "*.log"; do
        find . -type f -iname "${file}" | xargs "${RM}"
    done
}

function install_docker() {
    docker-compose up --build -d
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

function install_docker_deploy() {
    docker-compose up --build -d -f docker-compose-build.yml
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

function clean_docker() {
    docker stop $(docker ps -aq)
    docker system prune -a -f
    docker volume prune -f
}

function tags() {
    ctags -R .
    cscope -Rb
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
    '-s' | '--sync') sync_gpu_server ;;
        #
    '--tags') tags ;;
    '-h' | '--help') usage ;;
}

function sync_gpu_server() {
    rsync -av \
        --exclude-from=.rsync_ignore \
        ./src ./requirements.txt \
        xlapes02@sc-gpu1.fit.vutbr.cz:/home/xlapes02/ai-investing-remote
}

##### PARSE CLI-ARGS
[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
    '-c' | '--clean') clean ;;
    '-cd' | '--clean-docker') clean_docker ;;
    '-id' | '--install-docker') install_docker ;;
    '-idd' | '--install-docker-deploy') install_docker_deploy ;;
    '-dsip' | '--install-docker-deploy') docker_show_ipaddress ;;
        #
    '--create-samples-env') create_env_samples ;;
    '-s' | '--sync') sync_gpu_server ;;
        #
    '--tags') tags ;;
    '-h' | '--help') usage ;;
    esac
    shift
done
