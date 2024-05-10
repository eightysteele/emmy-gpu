################################################################################
FROM ghcr.io/nvidia/jax:t5x-2024-05-09 as base
################################################################################

# https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax

# See flags and environment variables:
# https://github.com/NVIDIA/JAX-Toolbox/blob/main/README.md#environment-variables

################################################################################
FROM base AS build-hlo
################################################################################

# Builds HLO
ARG URLREF_HLO=https://github.com/openxla/stablehlo.git#main
ARG SRC_PATH_HLO=/opt/hlo
RUN git-clone.sh ${URLREF_HLO} ${SRC_PATH_HLO}
ADD build-hlo.sh /usr/local/bin/
RUN build-hlo.sh

################################################################################
FROM build-hlo AS install-jdk
################################################################################

# Installs JDK
ARG SRC_PATH_HLO=/opt/hlo
COPY --from=build-hlo ${SRC_PATH_HLO} ${SRC_PATH_HLO}
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
FROM install-bazelisk AS build-xla-tools
################################################################################

# Builds XLA tooling: https://openxla.org/xla/tools
ARG SRC_PATH_XLA=/opt/xla
ADD build-xla-tools.sh /usr/local/bin
RUN build-xla-tools.sh

FROM build-xla-tools AS install-genjax
ARG GCLOUD_ACCESS_TOKEN
ADD install-genjax.sh /user/local/bin
RUN install-genjax.sh
