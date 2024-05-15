#!/bin/bash

BASE_IMAGE="pytorch/torchserve:latest-cpu"

DOCKER_TAG="pytorch/torchserve:chat_bot"

# Get relative path of example dir
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../../../..
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")

# Build docker image for the application
DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 --file ${EXAMPLE_DIR}/Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg EXAMPLE_DIR="${EXAMPLE_DIR}" --build-arg HUGGINGFACE_TOKEN -t "${DOCKER_TAG}" .

echo "Run the following command to start the chat bot"
echo ""
echo docker run --rm -it --platform linux/amd64 -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:8084:8084 -p 127.0.0.1:8085:8085 -v $(pwd)/model_store_1:/home/model-server/model-store -e MODEL_NAME="meta-llama/Llama-2-7b-chat-hf" $DOCKER_TAG
echo ""
echo "Note: You can replace the model identifier as needed"
