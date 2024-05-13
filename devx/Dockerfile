################################################################################
FROM ghcr.io/nvidia/jax:t5x-2024-05-09 as base
################################################################################

# https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax

# See flags and environment variables:
# https://github.com/NVIDIA/JAX-Toolbox/blob/main/README.md#environment-variables

################################################################################
FROM base AS build-llvm
################################################################################

# Builds LLVM/mlir
ARG SRC_PATH_LLVM=/opt/llvm
ADD build-mlir.sh /usr/local/bin/
RUN build-mlir.sh

################################################################################
FROM build-llvm AS install-jdk
################################################################################

# Installs JDK
ADD install-jdk.sh /usr/local/bin
RUN install-jdk.sh

################################################################################
FROM install-jdk AS install-clojure
################################################################################

# Installs Clojure
ADD install-clojure.sh /usr/local/bin
RUN install-clojure.sh

################################################################################
FROM install-clojure AS install-bazelisk
################################################################################

# Installs Bazel(isk)
ADD install-bazelisk.sh /usr/local/bin
RUN install-bazelisk.sh

################################################################################
#FROM install-bazelisk AS build-xla-tools
################################################################################

# Builds XLA tooling: https://openxla.org/xla/tools
# ARG SRC_PATH_XLA=/opt/xla
# ADD build-xla-tools.sh /usr/local/bin
# RUN build-xla-tools.sh

################################################################################
FROM install-bazelisk AS install-gcloud
################################################################################

# Installs gcloud
ARG GCLOUD_ACCESS_TOKEN
ADD install-gcloud.sh /usr/local/bin
RUN install-gcloud.sh

################################################################################
FROM install-gcloud AS install-genjax
################################################################################

# Installs GenJAX: https://github.com/probcomp/genjax
WORKDIR /
ARG GCLOUD_ACCESS_TOKEN
ADD genjax_demo.py /
ADD install-genjax.sh /usr/local/bin
RUN install-genjax.sh

################################################################################
FROM install-genjax AS final-image
################################################################################

ENV PATH="/opt/llvm/llvm-project/build/bin:${PATH}"
