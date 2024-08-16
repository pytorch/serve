ARG BASE_IMAGE=pytorch/torchserve:latest-cpu

FROM $BASE_IMAGE as server
ARG BASE_IMAGE
ARG EXAMPLE_DIR
ARG HUGGINGFACE_TOKEN
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV TS_DISABLE_TOKEN_AUTHORIZATION=true

USER root

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install libopenmpi-dev git -y

COPY $EXAMPLE_DIR/requirements.txt /home/model-server/chat_bot/requirements.txt
RUN pip install -r /home/model-server/chat_bot/requirements.txt
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

WORKDIR /home/model-server/chat_bot
RUN git clone -b int4_cpu_fix https://github.com/likholat/gpt-fast.git && \
    cd gpt-fast && \
    python -m pip install -r requirements.txt

COPY $EXAMPLE_DIR  /home/model-server/chat_bot
RUN cd sd && pip install -r requirements.txt

COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

WORKDIR /home/model-server/chat_bot
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
