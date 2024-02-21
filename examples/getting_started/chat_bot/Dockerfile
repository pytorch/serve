ARG BASE_IMAGE=pytorch/torchserve:latest-gpu

FROM $BASE_IMAGE as server
ARG BASE_IMAGE
ARG EXAMPLE_DIR
ARG MODEL_NAME_1
ARG MODEL_NAME_2
ARG HUGGINGFACE_TOKEN

USER root

ENV MODEL_NAME_1=$MODEL_NAME_1
ENV MODEL_NAME_2=$MODEL_NAME_2

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install libopenmpi-dev -y

COPY $EXAMPLE_DIR/requirements.txt /home/model-server/chat_bot/requirements.txt
RUN pip install -r /home/model-server/chat_bot/requirements.txt && huggingface-cli login --token $HUGGINGFACE_TOKEN
RUN pip uninstall torchtext torchdata torch torchvision torchaudio -y
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed
RUN pip uninstall torchserve torch-model-archiver torch-workflow-archiver -y
RUN pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly

COPY $EXAMPLE_DIR  /home/model-server/chat_bot
COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

WORKDIR /home/model-server/chat_bot
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
