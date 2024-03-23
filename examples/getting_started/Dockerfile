ARG BASE_IMAGE=pytorch/torchserve:latest-cpu

FROM $BASE_IMAGE as server
ARG BASE_IMAGE
ARG EXAMPLE_DIR
ARG HUGGINGFACE_TOKEN
ENV MODEL_NAME=$MODEL_NAME
ENV TORCH_COMPILE=$TORCH_COMPILE

USER root

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install jq wget -y


COPY $EXAMPLE_DIR/requirements.txt /home/model-server/getting_started/requirements.txt
RUN pip install -r /home/model-server/getting_started/requirements.txt

RUN \
    if echo "$MODEL_NAME" | grep -q "BERT"; then \
        huggingface-cli login --token $HUGGINGFACE_TOKEN; \
    fi

COPY $EXAMPLE_DIR  /home/model-server/getting_started
COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

WORKDIR /home/model-server/getting_started
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
