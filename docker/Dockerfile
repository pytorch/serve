# Final image for production
FROM pytorch/torchserve-runtime:latest AS runtime-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED TRUE

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
