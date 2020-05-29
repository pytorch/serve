#!/bin/bash

MACHINE=cpu
BRANCH_NAME="master"
DOCKER_TAG="pytorch/torchserve:latest"

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name to use"
          echo "-g, --gpu specify to use gpu"
          exit 0
          ;;
        -b|--branch_name)
          if test $
          then
            BRANCH_NAME="$2"
            shift
          else
            echo "Error! branch_name not provided"
            exit 1
          fi
          shift
          ;;
        -g|--gpu)
          MACHINE=gpu
          DOCKER_TAG="pytorch/torchserve:latest-gpu"
          shift
          ;;
	-t|--tag)
          DOCKER_TAG="$2"
          shift
          ;;
    esac
done

rm -rf serve
git clone https://github.com/pytorch/serve.git
cd serve
git checkout $BRANCH_NAME
cd ..
DOCKER_BUILDKIT=1 docker build --file Dockerfile_dev.$MACHINE -t $DOCKER_TAG .
