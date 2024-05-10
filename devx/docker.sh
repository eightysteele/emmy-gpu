#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EMMY_GPU_REPO=$(realpath "$(dirname "$SCRIPT_DIR")")
TAG=emmy-gpu:dev

check_gcloud() {
    echo "CLI: Checking for gcloud"
   if ! command -v gcloud > /dev/null; then
       ./install-gcloud.sh
   fi
   account=$(gcloud config list account --format="value(core.account)")
   if [ -z "$account" ]; then
       gcloud auth login
   else
       echo "CLI: gcloud authenticated with $account"
   fi
}

gcloud_auth_token() {
    echo "CLI: Getting gcloud token"
   if ! token=$(gcloud auth print-access-token); then
      echo "couldn't get gcloud auth token"
      exit 1
   fi
   echo "$token"
}

docker_build() {
    echo "CLI: Docker build"
    check_gcloud
    token=$(gcloud_auth_token)
    if ! docker build \
         --build-arg GCLOUD_ACCESS_TOKEN="$token" \
         --tag "$TAG" \
         --file Dockerfile .; then
        echo "docker build failed"
        exit 1
    fi

}

docker_run() {
    echo "CLI: Docker run"
    if ! sudo docker run \
         -v "$EMMY_GPU_REPO":/emmy-gpu \
         --shm-size=2g \
         --privileged \
         --gpus all \
         -it \
         "$TAG" \
         /bin/bash; then
        echo "docker run failed"
        exit 1
    fi
}

usage() {
    echo "-b docker build, -r docker run, -h help"
    exit 0
}

entrypoint() {
    cd "$SCRIPT_DIR"
    if ! cd "$SCRIPT_DIR"; then
        echo "failed to change into the project root directory $SCRIPT_DIR";
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
