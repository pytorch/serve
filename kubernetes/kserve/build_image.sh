#!/bin/bash

DOCKER_TAG="pytorch/torchserve-kfs:latest"
BASE_IMAGE="pytorch/torchserve:latest"
DOCKER_FILE="Dockerfile"

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-g, --gpu specify for gpu build"
          echo "-t, --tag specify tag name for docker image"
          exit 0
          ;;
        -g|--gpu)
          DOCKER_TAG="pytorch/torchserve-kfs:latest-gpu"
          BASE_IMAGE="pytorch/torchserve:latest-gpu"
          shift
          ;;
        -d|--dev)
          DOCKER_FILE="Dockerfile.dev"
          shift
          ;;
        -t|--tag)
          DOCKER_TAG="$2"
          shift
          shift
          ;;
    esac
done

cp ../../frontend/server/src/main/resources/proto/*.proto .

DOCKER_BUILDKIT=1 docker build --file "$DOCKER_FILE" --build-arg BASE_IMAGE=$BASE_IMAGE -t "$DOCKER_TAG" .
