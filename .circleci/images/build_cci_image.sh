#!/bin/bash

ENV_TYPE="pythn36"
BASE_IMAGE="ubuntu:18.04"

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-e, --env_type specify env_type to use from { pythn36, conda38, pyenv37, venv36, conda39 }"
          echo "-t, --tag specify tag name for docker image \"<image>:<tag>\""
          exit 0
          ;;
        -e|--env_type)
          ENV_TYPE="$2"
          shift
          shift
          ;;
        -t|--tag)
          DOCKER_TAG="$2"
          shift
          shift
          ;;
    esac
done

if [ ! "$DOCKER_TAG" ];
then
  DOCKER_TAG="pytorch/torchserve-build:ubuntu18-$ENV_TYPE-cpu"
fi

DOCKER_BUILDKIT=1 docker build -t $DOCKER_TAG --build-arg ENV_TYPE=$ENV_TYPE .

