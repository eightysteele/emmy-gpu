# syntax=docker/dockerfile:1.7-labs

# The build environment docs for javacpp-presets are incredibly out of date, so
# this Dockerfile roughly follows the deploy-ubuntu GitHub action instead:
#
# https://github.com/bytedeco/javacpp-presets/blob/master/.github/actions/deploy-ubuntu/action.yml

FROM --platform=linux/amd64 ubuntu:jammy AS build-init
LABEL authors="Aaron Steele <eightysteele@gmail.com>"
ARG ARCH=amd64
ARG ARCH_CUDA=x86_64
ARG PREFIX=x86_64-linux-gnu
ARG CUDA=cuda-repo-rhel8-12-3-local-12.3.2_545.23.08-1.x86_64.rpm
ARG CUDNN=8.9.7.29-1.cuda12.2.x86_64
ARG NCCL=2.19.3-1+cuda12.3.x86_64
ARG NVCOMP=nvcomp_3.0.5_x86_64_12.x
ARG MAVEN_HOME=/opt/maven
ARG CODENAME=jammy
ARG DEBIAN_FRONTEND=noninteractive
ENV TOOLCHAIN="gcc g++ gfortran linux-libc-dev-${ARCH}-cross binutils"
RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg \
    && dpkg --add-architecture ${ARCH} \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3 python2.7 python3-minimal python2.7-minimal rpm libasound2-dev:${ARCH} freeglut3-dev:${ARCH} libfontconfig-dev:${ARCH} libgtk2.0-dev:${ARCH} libusb-dev:${ARCH} libusb-1.0-0-dev:${ARCH} libffi-dev:${ARCH} libbz2-dev:${ARCH} zlib1g-dev:${ARCH} libxcb1-dev:${ARCH} libpulse-dev:${ARCH} \
    && apt-get -y install linux-libc-dev-amd64-cross pkg-config ccache clang $TOOLCHAIN openjdk-8-jdk ant python2 python3-pip swig git file wget unzip tar bzip2 gzip patch autoconf-archive autogen automake cmake make libtool bison flex perl nasm ragel curl libcurl4-openssl-dev libssl-dev libffi-dev libbz2-dev zlib1g-dev rapidjson-dev libdrm-dev

FROM build-init AS build-java-maven
WORKDIR /
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
RUN ln -sf $JAVA_HOME /usr/lib/jvm/default-java \
    && find /usr/lib/jvm/default-java/ \
    && curl -LO https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz \
    && curl -LO https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.18.3.tar.gz \
    && curl -LO https://services.gradle.org/distributions/gradle-7.1.1-bin.zip \
    && tar -xzf apache-maven-3.6.3-bin.tar.gz -C /opt/ \
    && tar -xzf git-2.18.3.tar.gz \
    && unzip gradle-7.1.1-bin.zip -d /opt/ \
    && ln -sf /opt/apache-maven-3.6.3/bin/mvn /usr/bin/mvn \
    && ln -sf /opt/gradle-7.1.1/bin/gradle /usr/bin/gradle \
    && python3 -m pip install gdown || python3 -m pip install gdown \
    && curl -s https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null \
    && curl -s -o /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-jammy.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list \
    && apt-get update && apt-get install -y libvulkan1 libvulkan-dev vulkan-sdk

FROM build-java-maven AS download-cuda
WORKDIR /
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
COPY --parents=true --from=build-java-maven /usr/lib/jvm/default-java /usr/lib/jvm/default-java
COPY --parents=true --from=build-java-maven /opt/apache-maven-3.6.3 /opt/apache-maven-3.6.3
COPY --parents=true --from=build-java-maven /opt/gradle-7.1.1 /opt/gradle-7.1.1
RUN mkdir cuda-downloads \
    && cd cuda-downloads/ \
    && curl -LO https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/${CUDA} \
    && curl -LO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH_CUDA}/libcudnn8-${CUDNN}.rpm \
    && curl -LO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH_CUDA}/libcudnn8-devel-${CUDNN}.rpm \
    && curl -LO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH_CUDA}/libnccl-${NCCL}.rpm \
    && curl -LO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH_CUDA}/libnccl-devel-${NCCL}.rpm

FROM download-cuda AS install-cuda
WORKDIR /
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
COPY --from=download-cuda /cuda-downloads /cuda-downloads
COPY --parents=true --from=download-cuda /usr/lib/jvm/default-java /usr/lib/jvm/default-java
COPY --parents=true --from=download-cuda /opt/apache-maven-3.6.3 /opt/apache-maven-3.6.3
COPY --parents=true --from=download-cuda /opt/gradle-7.1.1 /opt/gradle-7.1.1
RUN ln -sf $JAVA_HOME /usr/lib/jvm/default-java \
    && ln -sf /opt/apache-maven-3.6.3/bin/mvn /usr/bin/mvn \
    && ln -sf /opt/gradle-7.1.1/bin/gradle /usr/bin/gradle \
    && cd cuda-downloads/ \
    && rpm -i --force --ignorearch --nodeps ${CUDA} libcudnn*.rpm libnccl*.rpm \
    && rm -f *.rpm *.tgz *.txz *.tar.* \
    && cd /var/cuda-repo-rhel8-12-3-local \
    && rpm -i --force --ignorearch --nodeps cuda*.rpm libc*.rpm libn*.rpm \
    && rm *.rpm \
    && cd / \
    && ln -sf /usr/local/cuda/lib64/ /usr/local/cuda/lib \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
    && ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/libnvidia-ml.so \
    && mv /usr/include/cudnn* /usr/include/nccl* /usr/local/cuda/include/ \
    && mv /usr/lib64/libcudnn* /usr/lib64/libnccl* /usr/local/cuda/lib64/ \
    && curl -LO https://developer.download.nvidia.com/compute/nvcomp/3.0.5/local_installers/${NVCOMP}.tgz \
    && tar -xvf ${NVCOMP}.tgz -C /usr/local/cuda/lib64/ --strip-components=1 lib/ || tar -xvf ${NVCOMP}.tgz -C /usr/local/cuda/lib64/ --strip-components=2 nvcomp-3.0.5-ctk-12.2/lib/ \
    && tar -xvf ${NVCOMP}.tgz -C /usr/local/cuda/include/ --strip-components=1 include/ || tar -xvf ${NVCOMP}.tgz -C /usr/local/cuda/include/ --strip-components=2 nvcomp-3.0.5-ctk-12.2/include/ \
    && rm -f $NVCOMP.tgz \
    && mv /usr/include/cublas* /usr/include/nvblas* /usr/local/cuda/include/ || true \
    && mv /usr/lib64/libcublas* /usr/lib64/libnvblas* /usr/local/cuda/lib64/ || true \
    && for f in /usr/local/cuda/include/*_v8.h; do ln -sf $$f $${f:0:$${#f}-5}.h; done \
    && for f in /usr/local/cuda/lib64/libcudnn*so.8.*; do ln -sf $$f $${f:0:$${#f}-4};  ln -sf $$f $${f:0:$${#f}-6}; done \
    && for f in /usr/local/cuda/lib64/*.so.10; do ln -s $$f $$f.2; done \
    && for f in /usr/local/cuda/lib64/*.so.10; do ln -s $$f $${f:0:$${#f}-1}1; done \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.1 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.2 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.3 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.4 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.5 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.6 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.7 \
    && ln -s libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.8 \
    && cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so \
    && cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1 \
    && cp /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/lib64/libnvidia-ml.so \
    && cp /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/lib64/libnvidia-ml.so.1 \
    && sed -i s/cublas_v2.h/cublas_api.h/g /usr/local/cuda/include/cublasXt.h /usr/local/cuda/include/cusolverDn.h /usr/local/cuda/include/cusolverSp.h \
    && sed -i /block_merge_sort.cuh/d /usr/local/cuda/include/cub/cub.cuh \
    && sed -i /device_merge_sort.cuh/d /usr/local/cuda/include/cub/cub.cuh \
    && sed -i /device_segmented_sort.cuh/d /usr/local/cuda/include/cub/cub.cuh \
    && sed -i /warp_merge_sort.cuh/d /usr/local/cuda/include/cub/cub.cuh  \
    && rm -f $$(find /usr/local/cuda/ -name '*.a' -and -not -name libcudart_static.a -and -not -name libcudadevrt.a -and -not -name libnvcomp_device.a)
