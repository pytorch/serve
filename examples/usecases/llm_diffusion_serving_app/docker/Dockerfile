# Build arguments
ARG BASE_IMAGE=pytorch/torchserve:latest-cpu
FROM $BASE_IMAGE AS server

ARG EXAMPLE_DIR
ARG HUGGINGFACE_TOKEN
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Use multi-stage build with PyTorch Serve base image

# Set proxy environment variables if provided
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}
ENV TS_DISABLE_TOKEN_AUTHORIZATION=true

# Switch to root for installation
USER root

# Install system dependencies with cleanup in same layer
# RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
#     apt-get update && \
#     apt-get install -y \
#         libopenmpi-dev git \
#         && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install libopenmpi-dev git -y

WORKDIR /home/model-server/

# Copy and install main requirements
COPY ${EXAMPLE_DIR}/requirements.txt /home/model-server/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy and install SD-specific requirements
COPY ${EXAMPLE_DIR}/sd/requirements.txt /home/model-server/sd_requirements.txt
# RUN pip install --no-cache-dir -r sd_requirements.txt
RUN pip install -r sd_requirements.txt

# Copy application code
COPY ${EXAMPLE_DIR} /home/model-server/llm_diffusion_serving_app/

# Install requirements (now we can reference the files from the copied application)
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install --no-cache-dir -r /home/model-server/llm_diffusion_serving_app/requirements.txt && \
#     pip install --no-cache-dir -r /home/model-server/llm_diffusion_serving_app/sd/requirements.txt

# RUN pip install -r /home/model-server/llm_diffusion_serving_app/requirements.txt && \
#     pip install -r /home/model-server/llm_diffusion_serving_app/sd/requirements.txt

# Login to Hugging Face
RUN --mount=type=secret,id=hf_token \
    huggingface-cli login --token ${HUGGINGFACE_TOKEN}

WORKDIR /home/model-server/llm_diffusion_serving_app

COPY $EXAMPLE_DIR/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY $EXAMPLE_DIR/config.properties /home/model-server/config.properties

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server
