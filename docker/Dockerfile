# Final image for production
FROM pytorch/torchserve-base:latest AS runtime-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED TRUE

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
    # using openjdk-17-jdk due to circular dependency(ca-certificates) bug in openjdk-17-jre-headless debian package
    # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1009905
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

ENV PATH="/home/venv/bin:$PATH"

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
