#!/bin/bash

# Check if there are enough arguments
if [ "$#" -eq 0 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <HF Model 1> <HF Model 2>"
  exit 1
fi

MODEL_NAME_1=$(echo "$1" | sed 's/\//---/g')
MODEL_NAME_2=$(echo "$2" | sed 's/\//---/g')
echo "Models: " $MODEL_NAME_1 $MODEL_NAME_2

BASE_IMAGE="pytorch/torchserve:latest-gpu"

if command -v nvidia-smi &> /dev/null; then
    echo "Using TorchServe's GPU image"
    BASE_IMAGE="pytorch/torchserve:latest-gpu"
else
    echo "This example requires a CUDA enabled device"
    exit 1
fi

DOCKER_TAG="pytorch/torchserve:${MODEL_NAME_1}_${MODEL_NAME_2}"

# Get relative path of example dir
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../..
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")


# We need to build TorchServe image with CUDA runtime for using bitsandbytes quantization
cd docker
./build_image.sh  -bi nvidia/cuda:12.1.0-runtime-ubuntu20.04 -cv cu121 -t pytorch/torchserve:latest-gpu
cd ..

DOCKER_BUILDKIT=1 docker build --file ${EXAMPLE_DIR}/Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg EXAMPLE_DIR="${EXAMPLE_DIR}" --build-arg MODEL_NAME_1="${MODEL_NAME_1}" --build-arg MODEL_NAME_2="${MODEL_NAME_2}" --build-arg HUGGINGFACE_TOKEN -t "${DOCKER_TAG}" .

echo "Run the following command to start the chat bot"
echo ""
echo docker run --rm -it --gpus all -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 8084:8084 -p 8085:8085 -v $(pwd)/model_store_1:/home/model-server/model-store $DOCKER_TAG
echo ""
