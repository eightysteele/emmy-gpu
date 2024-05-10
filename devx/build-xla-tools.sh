#!/usr/bin/env bash

set -e

cd "$SRC_PATH_XLA"

bazelisk build //xla/tools:run_hlo_module
cp bazel-xla/bazel-out/k8-opt/bin/xla/tools/run_hlo_module /usr/local/bin

bazelisk build //xla/tools/multihost_hlo_runner:hlo_runner_main
cp bazel-xla/bazel-out/k8-opt/bin/xla/tools/multihost_hlo_runner/hlo_runner_main /usr/local/bin

bazelisk build //xla/tools:hlo-opt
cp bazel-xla/bazel-out/k8-opt/bin/xla/tools/hlo-opt /usr/local/bin
