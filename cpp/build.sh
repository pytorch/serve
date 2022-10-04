#!/usr/bin/env bash

# Obtain the base directory this script resides in.
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Useful constants
COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_OFF="\033[0m"

function detect_platform() {
  unameOut="$(uname -s)"
  case "${unameOut}" in
      Linux*)     PLATFORM=Linux;;
      Darwin*)    PLATFORM=Mac;;
      Windows*)   PLATFORM=Windows;;
      *)          PLATFORM="UNKNOWN:${unameOut}"
  esac
  echo -e "${COLOR_GREEN}Detected platform: $PLATFORM ${COLOR_OFF}"
}

function install_dependencies_linux() {
  sudo apt-get install -yq \
    autoconf \
    automake \

    git \
    cmake \
    m4 \
    g++ \
    flex \
    bison \
    libgflags-dev \
    google-mock \
    libkrb5-dev \
    libsasl2-dev \
    libnuma-dev \
    pkg-config \
    libssl-dev \
    libcap-dev \
    gperf \
    libevent-dev \
    libtool \
    libboost-all-dev \
    libjemalloc-dev \
    libsnappy-dev \
    wget \
    unzip \
    libiberty-dev \
    liblz4-dev \
    liblzma-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libsodium-dev \
    libdouble-conversion-dev \
    ninja-build
}

function install_dependencies_mac() {
  # install the default dependencies from homebrew
  brew install -f            \
    cmake                    \
    m4                       \
    boost                    \
    double-conversion        \
    gflags                   \
    gperf                    \
    libevent                 \
    lz4                      \
    snappy                   \
    xz                       \
    openssl                  \
    libsodium

  brew link                 \
    cmake                   \
    boost                   \
    double-conversion       \
    gflags                  \
    gperf                   \
    libevent                \
    lz4                     \
    snappy                  \
    openssl                 \
    xz                      \
    libsodium
}

function install_dependencies() {
  echo -e "${COLOR_GREEN}[ INFO ] install dependencies ${COLOR_OFF}"
  if [ "$PLATFORM" = "Linux" ]; then
    install_dependencies_linux
  elif [ "$PLATFORM" = "Mac" ]; then
    install_dependencies_mac
  else
    echo -e "${COLOR_RED}[ ERROR ] Unknown platform: $PLATFORM ${COLOR_OFF}"
    exit 1
  fi
}

function install_folly() {
  FOLLY_SRC_DIR=$BASE_DIR/third-party/folly
  FOLLY_BUILD_DIR=$DEPS_DIR/folly-build

  if [ ! -d "$FOLLY_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning folly repo ${COLOR_OFF}"
    git clone https://github.com/facebook/folly.git "$FOLLY_SRC_DIR"
    cd $FOLLY_SRC_DIR
    git checkout tags/v2022.06.27.00
  fi

  if [ ! -d "$FOLLY_BUILD_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Building Folly ${COLOR_OFF}"
    cd $FOLLY_SRC_DIR 

    if [ "$PLATFORM" = "Linux" ]; then
      SUDO="sudo"
    elif [ "$PLATFORM" = "Mac" ]; then
      SUDO=""
    fi
    $SUDO ./build/fbcode_builder/getdeps.py install-system-deps --recursive

    $SUDO ./build/fbcode_builder/getdeps.py build \
    --allow-system-packages \
    --scratch-path $FOLLY_BUILD_DIR \
    --extra-cmake-defines='{"CMAKE_CXX_FLAGS": "-fPIC -D_GLIBCXX_USE_CXX11_ABI=1"}'

    echo -e "${COLOR_GREEN}[ INFO ] Folly is installed ${COLOR_OFF}"
  fi

  cd "$BWD" || exit
  echo "$FOLLY_BUILD_DIR/installed"
}

function install_kineto() {
  if [ "$PLATFORM" = "Linux" ]; then
    echo -e "${COLOR_GREEN}[ INFO ] Skip install kineto on Linux ${COLOR_OFF}"
  elif [ "$PLATFORM" = "Mac" ]; then
    KINETO_SRC_DIR=$BASE_DIR/third-party/kineto

    if [ ! -d "$KINETO_SRC_DIR" ] ; then
      echo -e "${COLOR_GREEN}[ INFO ] Cloning kineto repo ${COLOR_OFF}"
      git clone --recursive https://github.com/pytorch/kineto.git "$KINETO_SRC_DIR"
      cd $KINETO_SRC_DIR/libkineto
      mkdir build && cd build
      cmake ..
      make install
    fi
  fi

  cd "$BWD" || exit
}

function install_libtorch() {
  if [ "$PLATFORM" = "Mac" ]; then
    echo -e "${COLOR_GREEN}[ INFO ] Skip install libtorch on Linux ${COLOR_OFF}"
  elif [ ! -d "$DEPS_DIR/libtorch" ] ; then
    cd "$DEPS_DIR" || exit
    if [ "$PLATFORM" = "Linux" ]; then
      echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Linux ${COLOR_OFF}"
      if [ "$CUDA" = "cu102" ]; then
        wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu102.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu102.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cu102.zip
      elif [ "$CUDA" = "cu113" ]; then
        wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip
      elif [ "$CUDA" = "cu116" ]; then
        wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu116.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cu116.zip
      else
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip
      fi
    elif [ "$PLATFORM" = "Windows" ]; then
      echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Mac ${COLOR_OFF}"
      # TODO: Windows
      echo -e "${COLOR_RED}[ ERROR ] Unknown platform: $PLATFORM ${COLOR_OFF}"
      exit 1
    fi 
    echo -e "${COLOR_GREEN}[ INFO ] libtorch is installed ${COLOR_OFF}"
  fi

  cd "$BWD" || exit
}

function build() {
  MAYBE_BUILD_QUIC=""
  if [ "$WITH_QUIC" == true ] ; then
    setup_mvfst
    MAYBE_BUILD_QUIC="-DBUILD_QUIC=On"
  fi

  MAYBE_USE_STATIC_DEPS=""
  MAYBE_LIB_FUZZING_ENGINE=""
  MAYBE_BUILD_SHARED_LIBS=""
  MAYBE_BUILD_TESTS="-DBUILD_TESTS=ON"
  if [ "$NO_BUILD_TESTS" == true ] ; then
    MAYBE_BUILD_TESTS="-DBUILD_TESTS=OFF"
  fi

  if [ -z "$PREFIX" ]; then
    PREFIX=$BWD
  fi

  MAYBE_CUDA_COMPILER=""
  if [ "$CUDA" != "" ]; then
    MAYBE_CUDA_COMPILER='-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc'
  fi

  # Build torchserve_cpp with cmake
  cd "$BWD" || exit
  FOLLY_CMAKE_DIR=$DEPS_DIR/folly-build/installed
  if [ "$PLATFORM" = "Linux" ]; then
    cmake                                                                                     \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR;$FOLLY_CMAKE_DIR;$DEPS_DIR/libtorch"                       \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"                                                          \
    "$MAYBE_BUILD_QUIC"                                                                       \
    "$MAYBE_BUILD_TESTS"                                                                      \
    "$MAYBE_BUILD_SHARED_LIBS"                                                                \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                                                               \
    "$MAYBE_USE_STATIC_DEPS"                                                                  \
    "$MAYBE_LIB_FUZZING_ENGINE"                                                               \
    "$MAYBE_CUDA_COMPILER"                                                                    \
    ..

    if [ "$CUDA" = "cu102" ] || [ "$CUDA" = "cu113" ] || [ "$CUDA" = "cu116" ]; then
      export LD_LIBRARY_PATHH=${LD_LIBRARY_PATH}:/usr/local/cuda/bin/nvcc
    fi
  elif [ "$PLATFORM" = "Mac" ]; then
    cmake                                                                                     \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$DEPS_DIR;$FOLLY_CMAKE_DIR;"   \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"                                                          \
    "$MAYBE_BUILD_QUIC"                                                                       \
    "$MAYBE_BUILD_TESTS"                                                                      \
    "$MAYBE_BUILD_SHARED_LIBS"                                                                \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                                                               \
    "$MAYBE_USE_STATIC_DEPS"                                                                  \
    "$MAYBE_LIB_FUZZING_ENGINE"                                                               \
    ..

    export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/opt/icu4c/lib
  else
    # TODO: Windows
    echo -e "${COLOR_RED}[ ERROR ] Unknown platform: $PLATFORM ${COLOR_OFF}"
    exit 1
  fi 
  
  make -j "$JOBS" 
  make install
  echo -e "${COLOR_GREEN}torchserve_cpp build is complete. To run unit test: \
  ./_build/test/torchserve_cpp_test ${COLOR_OFF}"

  if [ -f "$DEPS_DIR/../src/examples/libmnist_handler.dylib" ]; then
    mv $DEPS_DIR/../src/examples/libmnist_handler.dylib $DEPS_DIR/../../test/resources/torchscript_model/mnist/mnist_handler/libmnist_handler.dylib
  elif [ -f "$DEPS_DIR/../src/examples/libmnist_handler.so" ]; then
    mv $DEPS_DIR/../src/examples/libmnist_handler.so $DEPS_DIR/../../test/resources/torchscript_model/mnist/mnist_handler/libmnist_handler.so
  fi

  cd $DEPS_DIR/../..
  if [ -f "$DEPS_DIR/../test/torchserve_cpp_test" ]; then
    $DEPS_DIR/../test/torchserve_cpp_test
  else
    echo -e "${COLOR_RED}[ ERROR ] _build/test/torchserve_cpp_test not exist ${COLOR_OFF}"
    exit 1
  fi
}

# Parse args
JOBS=8
WITH_QUIC=false
INSTALL_DEPENDENCIES=false
PREFIX=""
COMPILER_FLAGS=""
CUDA=""
USAGE="./build.sh [-j num_jobs] [-g cu102|cu113|cu116] [-q|--with-quic] [-p|--prefix] [-x|--compiler-flags]"
while [ "$1" != "" ]; do
  case $1 in
    -j | --jobs ) shift
                  JOBS=$1
                  ;;
    -g | --cuda-version ) shift
                  CUDA=$1
                  ;;              
    -q | --with-quic )
                  WITH_QUIC=true
                  ;;
    --no-install-dependencies )
                  INSTALL_DEPENDENCIES=false
          ;;
    -t | --no-tests )
                  NO_BUILD_TESTS=true
      ;;
    -p | --prefix )
                  shift
                  PREFIX=$1
      ;;
    -x | --compiler-flags )
                  shift
                  COMPILER_FLAGS=$1
      ;;
    * )           echo $USAGE
                  exit 1
esac
shift
done

detect_platform

if [ "$INSTALL_DEPENDENCIES" == true ] ; then
  install_dependencies
fi

MAYBE_OVERRIDE_CXX_FLAGS=""
if [ -n "$COMPILER_FLAGS" ] ; then
  MAYBE_OVERRIDE_CXX_FLAGS="-DCMAKE_CXX_FLAGS=$COMPILER_FLAGS"
fi

BUILD_DIR=_build
if [ ! -d "$BUILD_DIR" ] ; then
  mkdir -p $BUILD_DIR
fi

set -e nounset
trap 'cd $BASE_DIR' EXIT
cd $BUILD_DIR || exit
BWD=$(pwd)
DEPS_DIR=$BWD/_deps
mkdir -p "$DEPS_DIR"

# Must execute from the directory containing this script
cd "$(dirname "$0")"

install_folly
install_kineto
install_libtorch
build
