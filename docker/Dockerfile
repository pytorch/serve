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

ARG CUDA_VERSION=""

RUN git clone --depth 1 https://github.com/pytorch/serve.git

WORKDIR "serve"

RUN \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            python ./ts_scripts/install_dependencies.py --cuda $CUDA_VERSION; \
        # Install the binary with the latest CPU image on a CUDA base image
        else \
            python ./ts_scripts/install_dependencies.py; \
        fi; \
    # Install the CPU binary
    else \
        python ./ts_scripts/install_dependencies.py; \
    fi

# Make sure latest version of torchserve is uploaded before running this
RUN python -m pip install --no-cache-dir torchserve torch-model-archiver torch-workflow-archiver

# Final image for production
FROM ${BASE_IMAGE} AS runtime-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
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
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY --chown=model-server --from=compile-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082 7070 7071

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
