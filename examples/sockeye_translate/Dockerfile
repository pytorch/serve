FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE

RUN useradd -m model-server && \
    mkdir -p /home/model-server/tmp

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    openjdk-8-jdk-headless \
    curl \
    vim && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install sockeye --no-deps -r requirements/sockeye/requirements.gpu-cu92.txt && \
    pip install --no-cache-dir mxnet-model-server && \
    pip install -r requirements/sockeye-serving/requirements.txt

COPY config/config.properties /home/model-server
COPY scripts/mms/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    chown -R model-server /home/model-server

EXPOSE 8080 8081

USER model-server
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="james.e.woo@gmail.com"
