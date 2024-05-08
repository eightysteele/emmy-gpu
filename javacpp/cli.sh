#!/usr/bin/env bash

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

docker_build() {
    echo "CLI: Docker build"
    if ! docker build --tag javacpp:dev --file Dockerfile .; then
        echo "docker build failed"
        exit 1
    fi
}

docker_run() {
    echo "CLI: Docker run"
    if ! sudo docker run --privileged --gpus all -it javacpp:dev /bin/bash; then
        echo "docker run failed"
        exit 1
    fi
}

usage() {
    echo "-b docker build, -r docker run, -h help"
    exit 0
}

entrypoint() {
    cd "$PROJECT_ROOT"
    if ! cd "$PROJECT_ROOT"; then
        echo "failed to change into the project root directory $PROJECT_ROOT";
        exit 1
    fi

    while getopts ":brh" opt; do
        case $opt in
            b)
                docker_build
                ;;
            r)
                docker_run
                ;;
            h)
                usage
                ;;
            \?)
                echo "Unknown option :("
                exit 1
                ;;
        esac
    done
}

entrypoint "$@"
