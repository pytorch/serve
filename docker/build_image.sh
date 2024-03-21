#!/bin/bash

set -o errexit -o nounset -o pipefail

MACHINE=cpu
BRANCH_NAME="master"
DOCKER_TAG="pytorch/torchserve:latest-cpu"
BUILD_TYPE="production"
BASE_IMAGE="ubuntu:20.04"
UPDATE_BASE_IMAGE=false
USE_CUSTOM_TAG=false
CUDA_VERSION=""
USE_LOCAL_SERVE_FOLDER=false
BUILD_WITH_IPEX=false
BUILD_CPP=false
BUILD_NIGHTLY=false
PYTHON_VERSION=3.9

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name to use"
          echo "-g, --gpu specify to use gpu"
          echo "-bi, --baseimage specify base docker image. Example: nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04 "
          echo "-bt, --buildtype specify for type of created image. Possible values: production, dev, ci."
          echo "-cv, --cudaversion specify to cuda version to use"
          echo "-t, --tag specify tag name for docker image"
          echo "-lf, --use-local-serve-folder specify this option for the benchmark image if the current 'serve' folder should be used during automated benchmarks"
          echo "-ipex, --build-with-ipex specify to build with intel_extension_for_pytorch"
          echo "-cpp, --build-cpp specify to build TorchServe CPP"
          echo "-py, --pythonversion specify to python version to use: Possible values: 3.8 3.9 3.10"
          echo "-n, --nightly specify to build with TorchServe nightly"
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
          BASE_IMAGE="nvidia/cuda:11.8.0-base-ubuntu20.04"
          CUDA_VERSION="cu117"
          shift
          ;;
        -bi|--baseimage)
          BASE_IMAGE="$2"
          UPDATE_BASE_IMAGE=true
          shift
          shift
          ;;
        -bt|--buildtype)
          BUILD_TYPE="$2"
          shift
          shift
          ;;
        -t|--tag)
          CUSTOM_TAG="$2"
          USE_CUSTOM_TAG=true
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
        -cpp|--build-cpp)
          BUILD_CPP=true
          shift
          ;;
        -n|--nightly)
          BUILD_NIGHTLY=true
          shift
          ;;
        -py|--pythonversion)
          PYTHON_VERSION="$2"
          if [[ $PYTHON_VERSION = 3.8 || $PYTHON_VERSION = 3.9 || $PYTHON_VERSION = 3.10 || $PYTHON_VERSION = 3.11 ]]; then
            echo "Valid python version"
          else
            echo "Valid python versions are 3.8, 3.9 3.10 and 3.11"
            exit 1
          fi
          shift
          shift
          ;;
        # With default ubuntu version 20.04
        -cv|--cudaversion)
          CUDA_VERSION="$2"
          if [ "${CUDA_VERSION}" == "cu121" ];
          then
            BASE_IMAGE="nvidia/cuda:12.1.0-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu118" ];
          then
            BASE_IMAGE="nvidia/cuda:11.8.0-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu117" ];
          then
            BASE_IMAGE="nvidia/cuda:11.7.1-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu116" ];
          then
            BASE_IMAGE="nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu113" ];
          then
            BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu111" ];
          then
            BASE_IMAGE="nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu102" ];
          then
            BASE_IMAGE="nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"
          elif [ "${CUDA_VERSION}" == "cu101" ]
          then
            BASE_IMAGE="nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
          elif [ "${CUDA_VERSION}" == "cu92" ];
          then
            BASE_IMAGE="nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04"
          else
            echo "CUDA version not supported"
            exit 1
          fi
          shift
          shift
          ;;
    esac
done

if [ "${MACHINE}" == "gpu" ] && $BUILD_WITH_IPEX ;
then
  echo "--gpu and --ipex are mutually exclusive. Please select one of them."
  exit 1
fi

if [ "${BUILD_TYPE}" == "dev" ] && ! $USE_CUSTOM_TAG ;
then
  if [ "${BUILD_CPP}" == "true" ]
  then
    DOCKER_TAG="pytorch/torchserve:cpp-dev-$MACHINE"
  else
    DOCKER_TAG="pytorch/torchserve:dev-$MACHINE"
  fi
fi

if [ "$USE_CUSTOM_TAG" = true ]
then
  DOCKER_TAG=${CUSTOM_TAG}
fi

if [[ $UPDATE_BASE_IMAGE == true && $MACHINE == "gpu" ]];
then
  echo "Incompatible options: -bi doesn't work with -g option"
  exit 1
fi

if [ "$BUILD_CPP" == "true" ];
then
  if [ "$BUILD_TYPE" != "dev" ];
  then
    echo "Only dev container build is supported for CPP"
    exit 1
  fi

  if [[ "${MACHINE}" == "gpu" || "${CUDA_VERSION}" != "" ]];
  then
    if [ "${CUDA_VERSION}" == "cu121" ];
    then
      BASE_IMAGE="nvidia/cuda:12.1.1-devel-ubuntu20.04"
    elif [ "${CUDA_VERSION}" == "cu118" ];
    then
      BASE_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu20.04"
    else
      echo "Cuda version $CUDA_VERSION is not supported for CPP"
      exit 1
    fi
  fi
fi

if [ "${BUILD_TYPE}" == "production" ]
then
  DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}"  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" --build-arg BUILD_NIGHTLY="${BUILD_NIGHTLY}" -t "${DOCKER_TAG}" --target production-image  .
elif [ "${BUILD_TYPE}" == "ci" ]
then
  DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}"  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" --build-arg BUILD_NIGHTLY="${BUILD_NIGHTLY}" --build-arg BRANCH_NAME="${BRANCH_NAME}"  -t "${DOCKER_TAG}" --target ci-image  .
else
  if [ "${BUILD_CPP}" == "true" ]
  then
    DOCKER_BUILDKIT=1 docker build --file Dockerfile.cpp --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}" --build-arg PYTHON_VERSION="${PYTHON_VERSION}" --build-arg BRANCH_NAME="${BRANCH_NAME}" -t "${DOCKER_TAG}" --target cpp-dev-image .
  else
    DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}"  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" --build-arg BUILD_NIGHTLY="${BUILD_NIGHTLY}" --build-arg BRANCH_NAME="${BRANCH_NAME}" --build-arg BUILD_WITH_IPEX="${BUILD_WITH_IPEX}"  -t "${DOCKER_TAG}" --target dev-image  .
  fi
fi
