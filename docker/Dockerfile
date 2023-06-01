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


ARG BASE_IMAGE=python:3.9-alpine

# Note:
# Define here the default python version to be used in all later build-stages as default.
# ARG and ENV variables do not persist across stages (they're build-stage scoped).
# That is crucial for ARG PYTHON_VERSION, which otherwise becomes "" leading to nasty bugs,
# that don't let the build fail, but break current version handling logic and result
# in images with wrong python version. To fix that, we will restate the ARG PYTHON_VERSION
# on each build-stage.
ARG PYTHON_VERSION=3.9

FROM ${BASE_IMAGE} AS compile-image


RUN apk --update upgrade

RUN apk add --no-cache \
    build-base \
    curl \
    openjdk17-jre-headless

RUN python -m pip install -U pip setuptools

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG CUDA_VERSION=""

RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-2])") && \
    TORCH_VISION_VER=$(curl --silent --location https://pypi.org/pypi/torchvision/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-2])") && \
    TORCH_TEXT_VER=$(curl --silent --location https://pypi.org/pypi/torchtext/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-2])") && \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            python -m pip install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION torchvision==$TORCH_VISION_VER+$CUDA_VERSION torchtext==$TORCH_TEXT_VER+$CUDA_VERSION -f https://download.pytorch.org/whl/torch_stable.html; \
        # Install the binary with the latest CUDA version support
        else \
            python -m pip install --no-cache-dir torch torchvision torchtext; \
        fi; \
        python -m pip install --no-cache-dir -r https://raw.githubusercontent.com/pytorch/serve/master/requirements/common.txt; \
    # Install the CPU binary
    else \
        python -m pip install --no-cache-dir torch==$TORCH_VER+cpu torchvision==$TORCH_VISION_VER+cpu torchtext==$TORCH_TEXT_VER+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    fi
RUN apk add --no-cache \
    linux-headers


RUN python -m pip install -U setuptools && python -m pip install --no-cache-dir captum torchserve torch-model-archiver
RUN apk add --no-cache \
    bash
RUN python -m pip install pyyaml


RUN adduser -D model-server  && mkdir -p /home/model-server/tmp
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
