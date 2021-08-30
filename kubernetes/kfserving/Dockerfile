# syntax = docker/dockerfile:experimental
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements

ARG BASE_IMAGE=pytorch/torchserve:latest
FROM ${BASE_IMAGE}

USER root

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
COPY kfserving_wrapper kfserving_wrapper
COPY config.properties config.properties

USER model-server

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]

