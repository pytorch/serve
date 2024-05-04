ARG BASE_IMAGE=pytorch/torchserve:latest-gpu

FROM $BASE_IMAGE as server
ARG BASE_IMAGE
ARG EXAMPLE_DIR
ARG HUGGINGFACE_TOKEN

USER root

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install libopenmpi-dev git -y

COPY $EXAMPLE_DIR/requirements.txt /home/model-server/chat_bot/requirements.txt
RUN pip install -r /home/model-server/chat_bot/requirements.txt && huggingface-cli login --token $HUGGINGFACE_TOKEN

WORKDIR /home/model-server/chat_bot
RUN git clone https://github.com/ggerganov/llama.cpp.git build && \
    cd build && \
    make && \
    python -m pip install -r requirements.txt

COPY $EXAMPLE_DIR  /home/model-server/chat_bot
COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

WORKDIR /home/model-server/chat_bot
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
