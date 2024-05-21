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
ARG GCC_VERSION=9
ARG BRANCH_NAME="master"
ARG USE_CUDA_VERSION=""

FROM ${BASE_IMAGE} AS cpp-dev-image
ARG BASE_IMAGE
ARG PYTHON_VERSION
ARG CMAKE_VERSION
ARG GCC_VERSION
ARG BRANCH_NAME
ARG USE_CUDA_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED TRUE
ENV TZ=Etc/UTC

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y \
        sudo \
        vim \
        git \
        curl \
        wget \
        rsync \
        gpg \
        gcc-$GCC_VERSION \
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

# Enable installation of recent cmake release and pin cmake & cmake-data version
# Ref: https://apt.kitware.com/
RUN (wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null) \
    && (echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null) \
    && apt-get update \
    && (test -f /usr/share/doc/kitware-archive-keyring/copyright || sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg) \
    && sudo apt-get install kitware-archive-keyring \
    && echo "Package: cmake\nPin: version $CMAKE_VERSION*\nPin-Priority: 1001" > /etc/apt/preferences.d/cmake \
    && echo "Package: cmake-data\nPin: version $CMAKE_VERSION*\nPin-Priority: 1001" > /etc/apt/preferences.d/cmake-data \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/pytorch/serve.git \
    && cd serve \
    && git checkout ${BRANCH_NAME}

WORKDIR "serve"

# CPP backend binary install depends on "ts" directory being present in python site-packages
RUN pip install pygit2 && python ts_scripts/install_from_src.py

EXPOSE 8080 8081 8082 7070 7071


FROM cpp-dev-image AS cpp-build-image
ARG BASE_IMAGE
ARG PYTHON_VERSION
ARG CMAKE_VERSION
ARG GCC_VERSION
ARG BRANCH_NAME
ARG USE_CUDA_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED TRUE
ENV TZ=Etc/UTC

RUN \
    if [ -n "$USE_CUDA_VERSION" ]; then \
        python ts_scripts/install_dependencies.py --cuda $USE_CUDA_VERSION --cpp --environment dev; \
    else \
        python ts_scripts/install_dependencies.py --cpp --environment dev; \
    fi

RUN cd cpp && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    make test


FROM ${BASE_IMAGE} AS cpp-prod-image
ARG BASE_IMAGE
ARG PYTHON_VERSION
ARG CMAKE_VERSION
ARG GCC_VERSION
ARG BRANCH_NAME
ARG USE_CUDA_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED TRUE
ENV TZ=Etc/UTC

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
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY --chown=model-server --from=cpp-build-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"
ENV LD_LIBRARY_PATH='python -c "import torch;from pathlib import Path;p=Path(torch.__file__);print(f\"{(p.parent / 'lib').as_posix()}:{(p.parents[1] / 'nvidia/nccl/lib').as_posix()}\")":$LD_LIBRARY_PATH'

COPY docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

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
