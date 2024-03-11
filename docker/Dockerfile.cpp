# syntax = docker/dockerfile:experimental
#
# This file can build images for CPU & GPU with CPP backend support.
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


ARG BASE_IMAGE=ubuntu:20.04
ARG PYTHON_VERSION=3.9
ARG CMAKE_VERSION=3.26.4
ARG BRANCH_NAME="master"
ARG USE_CUDA_VERSION=""

FROM ${BASE_IMAGE} AS cpp-dev-image
ARG BASE_IMAGE
ARG PYTHON_VERSION
ARG CMAKE_VERSION
ARG BRANCH_NAME
ARG USE_CUDA_VERSION
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        sudo \
        vim \
        git \
        curl \
        wget \
        rsync \
        gpg \
        ca-certificates \
        lsb-release \
        openjdk-17-jdk \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and "activate" it by adding it first to the path.
RUN python$PYTHON_VERSION -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# Enable installation of recent cmake release
# Ref: https://apt.kitware.com/
RUN (wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null) \
    && (echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null) \
    && apt-get update \
    && (test -f /usr/share/doc/kitware-archive-keyring/copyright || sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg) \
    && sudo apt-get install kitware-archive-keyring \
    && rm -rf /var/lib/apt/lists/*

# Pin cmake and cmake-data version
# Ref: https://manpages.ubuntu.com/manpages/xenial/man5/apt_preferences.5.html
RUN echo "Package: cmake\nPin: version $CMAKE_VERSION*\nPin-Priority: 1001" > /etc/apt/preferences.d/cmake
RUN echo "Package: cmake-data\nPin: version $CMAKE_VERSION*\nPin-Priority: 1001" > /etc/apt/preferences.d/cmake-data

# Install CUDA toolkit to enable "libtorch" build with GPU support
RUN apt-get update && \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        if [ "$USE_CUDA_VERSION" = "cu121" ]; then \
            apt-get -y install cuda-toolkit-12-1; \
        elif [ "$USE_CUDA_VERSION" = "cu118" ]; then \
            apt-get -y install cuda-toolkit-11-8; \
        else \
            echo "Cuda version not supported by CPP backend: $USE_CUDA_VERSION"; \
            exit 1; \
        fi; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/pytorch/serve.git \
    && cd serve \
    && git checkout ${BRANCH_NAME}

WORKDIR "serve"

# CPP backend binary install depends on "ts" directory being present in python site-packages
RUN pip install pygit2 && python ts_scripts/install_from_src.py

EXPOSE 8080 8081 8082 7070 7071
