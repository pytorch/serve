#!/bin/bash

BASE_IMAGE="pytorch/torchserve:latest-cpu"

DOCKER_TAG="pytorch/torchserve:ov_chat_bot"

# Get relative path of example dir
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../../../..
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")

# Build docker image for the application
DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 --file ${EXAMPLE_DIR}/Dockerfile \
--build-arg BASE_IMAGE="${BASE_IMAGE}" \
--build-arg EXAMPLE_DIR="${EXAMPLE_DIR}" \
--build-arg HUGGINGFACE_TOKEN \
--build-arg HTTP_PROXY=$http_proxy \
--build-arg HTTPS_PROXY=$https_proxy \
--build-arg NO_PROXY=$no_proxy \
 -t "${DOCKER_TAG}" .

mkdir -p model-store-local

echo ""
echo "Run the following command to start the chat bot"
echo ""
echo "docker run --rm -it --platform linux/amd64 \\
-p 127.0.0.1:8080:8080 \\
-p 127.0.0.1:8081:8081 \\
-p 127.0.0.1:8082:8082 \\
-p 127.0.0.1:8084:8084 \\
-p 127.0.0.1:8085:8085 \\
-v ${PWD}/model-store-local:/home/model-server/model-store \\
-e MODEL_NAME_LLM=meta-llama/Meta-Llama-3-8B-Instruct \\
-e MODEL_NAME_SD=stabilityai/stable-diffusion-xl-base-1.0 \\
$DOCKER_TAG"
echo ""
echo "Note: You can replace the model identifier as needed"
