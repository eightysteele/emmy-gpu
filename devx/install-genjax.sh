#!/usr/bin/env bash

set -e

apt-get update \
        apt-get install -y \
        curl \
        apt-transport-https \
        ca-certificates \
        gnupg

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

apt-get update \
    apt-get install -y \
    google-cloud-cli

pip install keyring keyrings.google-artifactregistry-auth

artifact_url=https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/

auth_url="https://oauth2accesstoken:$GCLOUD_ACCESS_TOKEN@$artifact_url"

pip install genjax --extra-index-url "$auth_url"

