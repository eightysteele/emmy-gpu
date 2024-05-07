#ARG MAVEN_VERSION=3.6.3
ARG MAVEN_VERSION=3.9.6
ARG MAVEN_HOME=/opt/maven

FROM --platform=linux/amd64 nvidia/cuda:12.4.0-devel-centos7 AS build-init

LABEL authors="Aaron Steele <eightysteele@gmail.com>"

RUN yum -y makecache fast && \
    yum -y update && \
    yum -y install centos-release-scl-rh epel-release && \
    yum install -y devtoolset-9 rh-git218 rh-java-common-ant \
    boost-devel ccache clang gcc-c++ gcc-gfortran java-1.8.0-openjdk-devel \
    ant python python3-devel python3-pip swig file which wget unzip tar \
    bzip2 gzip xz patch autoconf-archive automake make cmake3 libtool bison \
    flex perl-core nasm alsa-lib-devel freeglut-devel gtk2-devel libusb-devel \
    libusb1-devel curl-devel expat-devel gettext-devel openssl-devel \
    bzip2-devel zlib-devel SDL2-devel libva-devel libxkbcommon-devel \
    libxkbcommon-x11-devel xcb-util* fontconfig-devel libffi-devel ragel \
    ocl-icd-devel GeoIP-devel pcre-devel ssdeep-devel yajl-devel \
    llvm-toolset-7.0 && \
    yum clean all
WORKDIR /
RUN rm -rf /var/cache/yum

FROM build-init AS mvn-install
WORKDIR /
ARG MAVEN_VERSION=3.9.6
ARG MAVEN_DIR=apache-maven-${MAVEN_VERSION}
ARG MAVEN_TAR=${MAVEN_DIR}-bin.tar.gz
ARG MAVEN_URL=https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/${MAVEN_TAR}
RUN wget ${MAVEN_URL} -P /tmp/ && \
    tar xzf /tmp/${MAVEN_TAR} -C /opt/ && \
    ln -s /opt/${MAVEN_DIR} /opt/maven

# We need the latest Maven version which isn't availble in the CentOS registry
FROM mvn-install AS javacpp-install
WORKDIR /
ARG MAVEN_HOME=/opt/maven
COPY --from=mvn-install ${MAVEN_HOME} ${MAVEN_HOME}
ENV PATH=${MAVEN_HOME}/bin:${PATH}
RUN git clone https://github.com/bytedeco/javacpp.git && \
    cd javacpp/ && \
    source scl_source enable llvm-toolset-7.0 devtoolset-9 rh-git218 rh-java-common && \
    mvn install -Djavacpp.platform=linux-x86_64

FROM javacpp-install AS javacpp-presets-install
WORKDIR /
ARG MAVEN_HOME=/opt/maven
ARG JAVACPP_HOME=/javacpp
COPY --from=javacpp-install ${MAVEN_HOME} ${MAVEN_HOME}
COPY --from=javacpp-install ${JAVACPP_HOME} ${JAVACPP_HOME}
ENV PATH=${MAVEN_HOME}/bin:${PATH}
RUN git clone https://github.com/bytedeco/javacpp-presets.git && \
    cd javacpp-presets/ && \
    source scl_source enable llvm-toolset-7.0 devtoolset-9 rh-git218 rh-java-common && \
    mvn install -Djavacpp.platform=linux-x86_64
