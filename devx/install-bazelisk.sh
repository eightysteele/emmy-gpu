#!/usr/bin/env bash

set -e

curl -L -O https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64

mv bazelisk-linux-amd64 /usr/local/bin/bazelisk

chmod a+x /usr/local/bin/bazelisk

bazelisk

