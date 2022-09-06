#!/bin/bash

###############################################################################
# BUG: ---------------------- NOT MAINTAINED ANYMORE --------------------------
###############################################################################

#set -x # debug

complete -c timedatectl -s h -l help -d 'Print a short help text and exit'

RM="rm -rfd"
RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'
PROJECT_NAME="diagram-editor"
ZIP_NAME="xlapes02"

##### FUNCTIONS
function error_exit() {
    printf "${RED}ERROR: $1${NC}\n"
    usage
    exit 1
}

function install() {
    python3 -m venv venv || error_exit "venv creating"
    source venv/bin/activate || error_exit "venv activation"
    pip3 install -r requirements.txt
    pip3 install -e .
    deactivate
}

function clean() {
    ${RM} *.zip
    find * -type d -iname "venv" | xargs ${RM}
    find * -type d -iname "__pycache__" | xargs ${RM}
    find * -type f -iname ".DS_Store" | xargs ${RM}
    ${RM} tags
    #    echo "${RM} .idea"
    #    echo "${RM} .cache"
}

function zip_project() {
    zip -r ${ZIP_NAME}.zip src/* requirements.txt start.sh SUR_projekt2021-2022 .gitmodules *.ipynb dataset/* debug/*
}

function ssh() {
    zip_project
    scp "$(pwd)/${ZIP_NAME}.zip" $1@eva.fit.vutbr.cz:/homes/eva/xl/$1
}

function line_of_codes() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        (brew list | grep "cloc") && cloc src/** # cloc src/**/*.{py,yaml}
    elif [[ "$(uname -s)" == "Linux" ]]; then
        echo "$(uname -s)"
    else
        echo "$(uname -s)"
    fi

}

function tags() {
    ctags -R .
    cscope -Rb
}

function usage() {
    echo "USAGE:
    '-r' | '--run') run ;;
    '-c' | '--clean') clean ;;
    '-z' | '--zip') zip_project ;;
    '-sz' | '--ssh-zdenek') ssh 'xlapes02' ;;
    '--cloc') line_of_codes ;;
    '--tags') tags ;;
    '-h' | '--help' | *) usage ;;"
}

function run() {
    source venv/bin/activate || error_exit "venv"
    cd src || error_exit "cd cmd..."
    python3 main.py -hp hyperparams.yaml -s
    cd .. || error_exit "cd cmd..."
}

##### PARSE CLI-ARGS
[[ "$#" -eq 0 ]] && usage && exit 0
while [ "$#" -gt 0 ]; do
    case "$1" in
#    '-r' | '--run') run ;;
    '-i' | '--install') install ;;
    '-c' | '--clean') clean ;;
    '-rs' | '--run-stock') run ;;
    '-h' | '--help' | *) usage ;;
    esac
    shift
done
