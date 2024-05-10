#!/usr/bin/env bash

set -e

print_var() {
    echo "$1: ${!1}"
}

export LLVM_ENABLED_LLD="ON"

print_var LLVM_ENABLED_LLD

apt-get update
apt-get install -y \
        ninja-build

[[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" || LLVM_ENABLE_LLD="OFF"

cd "$SRC_PATH_HLO" && git clone https://github.com/llvm/llvm-project.git

(cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))

MLIR_ENABLE_BINDINGS_PYTHON=OFF build_tools/build_mlir.sh "${PWD}"/llvm-project/ "${PWD}"/llvm-build

mkdir -p build && cd build

cmake .. -GNinja \
      -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
      -DMLIR_DIR="${PWD}"/../llvm-build/lib/cmake/mlir

cmake --build .

ninja check-stablehlo-tests

