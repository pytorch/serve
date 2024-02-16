ARG BASE_IMAGE=pytorch/torchserve:latest-gpu

FROM $BASE_IMAGE as server
ARG BASE_IMAGE
ARG EXAMPLE_DIR
ARG MODEL_NAME
ARG HUGGINGFACE_TOKEN

USER root

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install libopenmpi-dev -y

COPY $EXAMPLE_DIR/requirements.txt /home/model-server/$MODEL_NAME/requirements.txt
RUN pip install -r /home/model-server/$MODEL_NAME/requirements.txt && huggingface-cli login --token $HUGGINGFACE_TOKEN
COPY $EXAMPLE_DIR  /home/model-server/$MODEL_NAME
COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

WORKDIR /home/model-server/$MODEL_NAME
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
