#!/bin/bash

MACHINE=cpu
BRANCH_NAME="master"
DOCKER_TAG="pytorch/torchserve:latest-cpu"
BUILD_TYPE="production"
DOCKER_FILE="Dockerfile"
BASE_IMAGE="ubuntu:18.04"
CUSTOM_TAG=false
CUDA_VERSION=""
UBUNTU_VERSION="ubuntu:18.04"
USE_LOCAL_SERVE_FOLDER=false
BUILD_WITH_IPEX=false

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name to use"
          echo "-g, --gpu specify to use gpu"
          echo "-bt, --buildtype specify to created image for codebuild. Possible values: production, dev, codebuild."
          echo "-cv, --cudaversion specify to cuda version to use"
          echo "-ub, --ubuntu specify ubuntu version. Possible values: ubuntu:18.04, ubuntu 20.04"
          echo "-t, --tag specify tag name for docker image"
          echo "-lf, --use-local-serve-folder specify this option for the benchmark image if the current 'serve' folder should be used during automated benchmarks"
          echo "-ipex, --build-with-ipex specify to build with intel_extension_for_pytorch"
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
          BASE_IMAGE="nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"
          CUDA_VERSION="cu102"
          shift
          ;;
        -bt|--buildtype)
          BUILD_TYPE="$2"
          shift
          shift
          ;;
        -t|--tag)
          DOCKER_TAG="$2"
          CUSTOM_TAG=true
          shift
          shift
          ;;
        -lf|--use-local-serve-folder)
          USE_LOCAL_SERVE_FOLDER=true
          shift
          ;;
        -ipex|--build-with-ipex)
          BUILD_WITH_IPEX=true
          shift
          ;;
        # With default ubuntu version 18.04
        -cv|--cudaversion)
          CUDA_VERSION="$2"
          if [ $CUDA_VERSION == "cu116" ];
          then
            BASE_IMAGE="nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu113" ];
          then
            BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu111" ];
          then
            BASE_IMAGE="nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu102" ];
          then
            BASE_IMAGE="nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu101" ]
          then
            BASE_IMAGE="nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu92" ];
          then
            BASE_IMAGE="nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04"
          else
            echo "CUDA version not supported"
            exit 1
          fi
          shift
          shift
          ;;
        # CUDA 10 is not supported on Ubuntu 20.04
        -ub|--ubuntu)
          UBUNTU_VERSION="$2"
          if [[ $CUDA_VERSION == "cu116"  &&  $UBUNTU_VERSION == "ubuntu20.04" ]];
          then
            BASE_IMAGE="nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04"
          elif [[ $CUDA_VERSION == "cu113" && $UBUNTU_VERSION == "ubuntu20.04" ]];
          then
            BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
          elif [[ $CUDA_VERSION == "cu111" && $UBUNTU_VERSION == "ubuntu20.04" ]];
          then
            BASE_IMAGE="nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04"
          elif [[ $UBUNTU_VERSION == "ubuntu20.04" ]];
          then
            echo "Using CPU image"
            BASE_IMAGE="ubuntu:20.04"
          else
            echo "Ubuntu and CUDA version combination is not supported"
            echo $UBUNTU_VERSION
            echo $CUDA_VERSION
            exit 1
          fi
          shift
          ;;
    esac
done

if [ "${MACHINE}" == "gpu" ] && $BUILD_WITH_IPEX ;
then
  echo "--gpu and --ipex are mutually exclusive. Please select one of them."
  exit 1
fi

if [ "${BUILD_TYPE}" == "dev" ] && ! $CUSTOM_TAG ;
then
  DOCKER_TAG="pytorch/torchserve:dev-$MACHINE"
fi

if [ "${BUILD_TYPE}" == "codebuild" ] && ! $CUSTOM_TAG ;
then
  DOCKER_TAG="pytorch/torchserve:codebuild-$MACHINE"
fi

if [ $BUILD_TYPE == "production" ]
then
  DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg CUDA_VERSION=$CUDA_VERSION -t $DOCKER_TAG .
elif [ $BUILD_TYPE == "benchmark" ]
then
  DOCKER_BUILDKIT=1 docker build --pull --no-cache --file Dockerfile.benchmark --build-arg USE_LOCAL_SERVE_FOLDER=$USE_LOCAL_SERVE_FOLDER --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BRANCH_NAME=$BRANCH_NAME --build-arg CUDA_VERSION=$CUDA_VERSION --build-arg MACHINE_TYPE=$MACHINE -t $DOCKER_TAG .
else
  DOCKER_BUILDKIT=1 docker build --pull --no-cache --file Dockerfile.dev -t $DOCKER_TAG --build-arg BUILD_TYPE=$BUILD_TYPE --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BRANCH_NAME=$BRANCH_NAME --build-arg CUDA_VERSION=$CUDA_VERSION --build-arg MACHINE_TYPE=$MACHINE --build-arg BUILD_WITH_IPEX=$BUILD_WITH_IPEX .
fi
