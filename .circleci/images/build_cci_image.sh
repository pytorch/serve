#!/bin/bash

ENV_TYPE="py36"
BASE_IMAGE="ubuntu:18.04"

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-e, --env_type specify env_type to use from {py36, venv36, pyenv37, conda38}"
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
  DOCKER_TAG="pytorch/torchserve-cci:$ENV_TYPE"
fi

DOCKER_BUILDKIT=1 docker build -t $DOCKER_TAG --build-arg ENV_TYPE=$ENV_TYPE .

