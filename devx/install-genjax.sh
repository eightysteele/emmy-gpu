#!/usr/bin/env bash

set -e

pip install keyring keyrings.google-artifactregistry-auth

url=us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple

auth_url=https://oauth2accesstoken:"$GCLOUD_ACCESS_TOKEN"@"$url"

pip install genjax --extra-index-url "$auth_url"
