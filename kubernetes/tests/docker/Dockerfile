FROM pytorch/torchserve-nightly:latest-gpu
ARG EXAMPLE_DIR
USER root

RUN apt-get update && apt-get install jq -y
COPY $EXAMPLE_DIR/../docker/config.properties /home/model-server/config.properties
USER model-server
