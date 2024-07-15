# syntax = docker/dockerfile:experimental
#
# This file can build images for cpu and gpu env. By default it builds image for CPU.
# Use following option to build image for cuda/GPU: --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Here is complete command for GPU/cuda -
# $ DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/


ARG BASE_IMAGE=ubuntu:rolling

# Note:
# Define here the default python version to be used in all later build-stages as default.
# ARG and ENV variables do not persist across stages (they're build-stage scoped).
# That is crucial for ARG PYTHON_VERSION, which otherwise becomes "" leading to nasty bugs,
# that don't let the build fail, but break current version handling logic and result
# in images with wrong python version. To fix that, we will restate the ARG PYTHON_VERSION
# on each build-stage.
ARG PYTHON_VERSION=3.9

FROM ${BASE_IMAGE} AS compile-image
ARG BASE_IMAGE=ubuntu:rolling
ARG PYTHON_VERSION
ARG BUILD_NIGHTLY
ARG BUILD_FROM_SRC
ARG LOCAL_CHANGES
ARG BRANCH_NAME
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        g++ \
        python3-distutils \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
        openjdk-17-jdk \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Make the virtual environment and "activating" it by adding it first to the path.
# From here on the python$PYTHON_VERSION interpreter is used and the packages
# are installed in /home/venv which is what we need for the "runtime-image"
RUN python$PYTHON_VERSION -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

RUN python -m pip install -U pip setuptools

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG USE_CUDA_VERSION=""

COPY ./ serve

RUN \
    if echo "$LOCAL_CHANGES" | grep -q "false"; then \
        rm -rf serve;\
        git clone --recursive https://github.com/pytorch/serve.git -b $BRANCH_NAME; \
    fi


WORKDIR "serve"

RUN cp docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$USE_CUDA_VERSION" ]; then \
            python ./ts_scripts/install_dependencies.py --cuda $USE_CUDA_VERSION;\
        # Install the binary with the latest CPU image on a CUDA base image
        else \
            python ./ts_scripts/install_dependencies.py;\
        fi; \
    # Install the CPU binary
    else \
        python ./ts_scripts/install_dependencies.py; \
    fi

# Make sure latest version of torchserve is uploaded before running this
RUN \
    if echo "$BUILD_FROM_SRC" | grep -q "true"; then \
        python -m pip install -r requirements/developer.txt;\
        python ts_scripts/install_from_src.py;\
    elif echo "$BUILD_NIGHTLY" | grep -q "false"; then \
        python -m pip install --no-cache-dir torchserve torch-model-archiver torch-workflow-archiver;\
    else \
        python -m pip install --no-cache-dir torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly;\
    fi

# Final image for production
FROM ${BASE_IMAGE} AS production-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python$PYTHON_VERSION \
    python3-distutils \
    python$PYTHON_VERSION-dev \
    python$PYTHON_VERSION-venv \
    # using openjdk-17-jdk due to circular dependency(ca-certificates) bug in openjdk-17-jre-headless debian package
    # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1009905
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY --chown=model-server --from=compile-image /home/venv /home/venv
COPY --from=compile-image /usr/local/bin/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
ENV PATH="/home/venv/bin:$PATH"

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

COPY docker/config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082 7070 7071

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

# Final image for docker regression
FROM ${BASE_IMAGE} AS ci-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ARG BRANCH_NAME
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python$PYTHON_VERSION \
    python3-distutils \
    python$PYTHON_VERSION-dev \
    python$PYTHON_VERSION-venv \
    # using openjdk-17-jdk due to circular dependency(ca-certificates) bug in openjdk-17-jre-headless debian package
    # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1009905
    openjdk-17-jdk \
    build-essential \
    wget \
    numactl \
    nodejs \
    npm \
    zip \
    unzip \
    && npm install -g newman@5.3.2 newman-reporter-htmlextra markdown-link-check \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp


COPY --from=compile-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN python -m pip install --no-cache-dir -r https://raw.githubusercontent.com/pytorch/serve/$BRANCH_NAME/requirements/developer.txt

RUN mkdir /home/serve
ENV TS_RUN_IN_DOCKER True

WORKDIR /home/serve
CMD ["python", "test/regression_tests.py"]

#Final image for developer Docker image
FROM ${BASE_IMAGE} as dev-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ARG BRANCH_NAME
ARG BUILD_FROM_SRC
ARG LOCAL_CHANGES
ARG BUILD_WITH_IPEX
ARG IPEX_VERSION=1.11.0
ARG IPEX_URL=https://software.intel.com/ipex-whl-stable
ENV PYTHONUNBUFFERED TRUE
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    sudo \
    g++ \
    git \
    python$PYTHON_VERSION \
    python$PYTHON_VERSION-dev \
    python3-distutils \
    python$PYTHON_VERSION-venv \
    # using openjdk-17-jdk due to circular dependency(ca-certificates) bug in openjdk-17-jre-headless debian package
    # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1009905
    openjdk-17-jdk \
    build-essential \
    curl \
    vim \
    numactl \
    && if [ "$BUILD_WITH_IPEX" = "true" ]; then apt-get update && apt-get install -y libjemalloc-dev libgoogle-perftools-dev libomp-dev && ln -s /usr/lib/x86_64-linux-gnu/libjemalloc.so /usr/lib/libjemalloc.so && ln -s /usr/lib/x86_64-linux-gnu/libtcmalloc.so /usr/lib/libtcmalloc.so && ln -s /usr/lib/x86_64-linux-gnu/libiomp5.so /usr/lib/libiomp5.so; fi \
    && rm -rf /var/lib/apt/lists/*

COPY ./ serve

RUN \
    if echo "$LOCAL_CHANGES" | grep -q "false"; then \
        rm -rf serve;\
        git clone --recursive https://github.com/pytorch/serve.git -b $BRANCH_NAME; \
    fi

COPY --from=compile-image /home/venv /home/venv
ENV PATH="/home/venv/bin:$PATH"
WORKDIR "serve"
RUN python -m pip install -U pip setuptools \
    && python -m pip install --no-cache-dir -r requirements/developer.txt \
    && python ts_scripts/install_from_src.py \
    && useradd -m model-server \
    && mkdir -p /home/model-server/tmp \
    && cp docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh \
    && chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server \
    && cp docker/config.properties /home/model-server/config.properties \
    && mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store \
    && chown -R model-server /home/venv
EXPOSE 8080 8081 8082 7070 7071
USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
