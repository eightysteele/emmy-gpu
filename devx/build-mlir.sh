#!/usr/bin/env bash

set -e

apt-get update
apt-get install -y ninja-build

mkdir "$SRC_PATH_LLVM" && cd "$SRC_PATH_LLVM"
git clone https://github.com/llvm/llvm-project.git

cd llvm-project
mkdir build && cd build

cmake -GNinja ../llvm \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_PLATFORM_NO_VERSIONED_SONAME:BOOL=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_LLD=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_INCLUDE_TOOLS=ON \
      -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_VERSION_SUFFIX="" \
      -DLLVM_BUILD_TOOLS=ON \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_USE_SPLIT_DWARF=ON \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_USE_SANITIZER="Address;Undefined" \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DMLIR_BUILD_EXAMPLES=ON \
      -DMLIR_INCLUDE_TOOLS=ON

cmake --build . --target all
