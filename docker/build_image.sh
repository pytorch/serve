#!/bin/bash

MACHINE=cpu
BRANCH_NAME="master"
DOCKER_TAG="pytorch/torchserve:dev-cpu"
BUILD_TYPE="dev"
BASE_IMAGE="ubuntu:18.04"
CUSTOM_TAG=false
CUDA_VERSION=latest

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name to use"
          echo "-g, --gpu specify to use gpu"
          echo "-c, --codebuild specify to created image for codebuild"
          echo "-t, --tag specify tag name for docker image"
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
          DOCKER_TAG="pytorch/torchserve:dev-gpu"
          BASE_IMAGE="nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
          shift
          ;;
        -c|--codebuild)
          BUILD_TYPE="codebuild"
          shift
          ;;
        -t|--tag)
          DOCKER_TAG="$2"
          CUSTOM_TAG=true
          shift
          shift
          ;;
        -cv|--cudaversion)
          CUDA_VERSION="$2"
          shift
          shift
          ;;
    esac
done

if [ "${BUILD_TYPE}" == "codebuild" ] && ! $CUSTOM_TAG ;
then
  DOCKER_TAG="pytorch/torchserve:codebuild-$MACHINE"
fi

DOCKER_BUILDKIT=1 docker build --file Dockerfile.dev -t $DOCKER_TAG --build-arg BUILD_TYPE=$BUILD_TYPE --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BRANCH_NAME=$BRANCH_NAME --build-arg CUDA_VERSION=$CUDA_VERSION --build-arg MACHINE_TYPE=$MACHINE  .
