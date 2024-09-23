#!/bin/bash

MACHINE=cpu
DOCKER_TAG="pytorch/torchserve-kfs:latest"
BASE_IMAGE="pytorch/torchserve:latest-cpu"
DOCKER_FILE="Dockerfile"
BUILD_NIGHTLY=false
USE_CUSTOM_TAG=false
ARCH="linux/arm64,linux/amd64"
MULTI=false

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-g, --gpu specify for gpu build"
          echo "-t, --tag specify tag name for docker image"
          echo "-n, --nightly specify to build with TorchServe nightly"
          exit 0
          ;;
        -g|--gpu)
          MACHINE=gpu
          DOCKER_TAG="pytorch/torchserve-kfs:latest-gpu"
          BASE_IMAGE="pytorch/torchserve:latest-gpu"
          shift
          ;;
        -d|--dev)
          DOCKER_FILE="Dockerfile.dev"
          shift
          ;;
        -n|--nightly)
          BUILD_NIGHTLY=true
          shift
          ;;
        -t|--tag)
          CUSTOM_TAG="$2"
          USE_CUSTOM_TAG=true
          shift
          shift
          ;;
        -m|--multi)
          MULTI=true
          shift
          ;;
    esac
done

if [[ "${MACHINE}" == "gpu" ]] && [[ "$BUILD_NIGHTLY" == true ]] ;
then
  BASE_IMAGE="pytorch/torchserve-nightly:latest-gpu"
elif [[ "${MACHINE}" == "cpu" ]] && [[ "$BUILD_NIGHTLY" == true ]] ;
then
  BASE_IMAGE="pytorch/torchserve-nightly:latest-cpu"
fi

if [ "$USE_CUSTOM_TAG" = true ]
then
  DOCKER_TAG=${CUSTOM_TAG}
fi

cp ../../frontend/server/src/main/resources/proto/*.proto .
cp -r ../../third_party .

if [ "${MULTI}" == "true" ]; then
  DOCKER_BUILDKIT=1 docker buildx build --file "$DOCKER_FILE" --build-arg BASE_IMAGE=$BASE_IMAGE --platform "${ARCH}" -t "$DOCKER_TAG" --push .
else
  DOCKER_BUILDKIT=1 docker buildx build --file "$DOCKER_FILE" --build-arg BASE_IMAGE=$BASE_IMAGE -t "$DOCKER_TAG" --push .
fi
