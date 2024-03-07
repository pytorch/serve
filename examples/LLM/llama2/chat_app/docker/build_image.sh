#!/bin/bash

# Check if there are enough arguments
if [ "$#" -eq 0 ] || [ "$#" -gt 1 ]; then
  echo "Usage: $0 <HF Model>"
  exit 1
fi

MODEL_NAME=$(echo "$1" | sed 's/\//---/g')
echo "Model: " $MODEL_NAME

BASE_IMAGE="pytorch/torchserve:latest-cpu"

DOCKER_TAG="pytorch/torchserve:${MODEL_NAME}"

# Get relative path of example dir
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../../../..
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")

# Build docker image for the application
DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 --file ${EXAMPLE_DIR}/Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg EXAMPLE_DIR="${EXAMPLE_DIR}" --build-arg MODEL_NAME="${MODEL_NAME}"  --build-arg HUGGINGFACE_TOKEN -t "${DOCKER_TAG}" .

echo "Run the following command to start the chat bot"
echo ""
echo docker run --rm -it --platform linux/amd64 -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:8084:8084 -p 127.0.0.1:8085:8085 -v $(pwd)/model_store_1:/home/model-server/model-store $DOCKER_TAG
echo ""
