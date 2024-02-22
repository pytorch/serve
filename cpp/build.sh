#!/usr/bin/env bash

# Obtain the base directory this script resides in.
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "BASE_DIR=${BASE_DIR}"

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

function install_folly() {
  FOLLY_SRC_DIR=$BASE_DIR/third-party/folly
  FOLLY_BUILD_DIR=$DEPS_DIR/folly-build

  if [ ! -d "$FOLLY_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning folly repo ${COLOR_OFF}"
    git clone https://github.com/facebook/folly.git "$FOLLY_SRC_DIR"
    cd $FOLLY_SRC_DIR
    git checkout tags/v2024.01.29.00
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
  TORCH_VERSION="2.2.0"
  if [ "$PLATFORM" = "Mac" ]; then
    if [ ! -d "$DEPS_DIR/libtorch" ]; then
      if [[ $(uname -m) == 'x86_64' ]]; then
        echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Mac x86_64 ${COLOR_OFF}"
        wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${TORCH_VERSION}.zip
        unzip libtorch-macos-x86_64-${TORCH_VERSION}.zip
        rm libtorch-macos-x86_64-${TORCH_VERSION}.zip
      else
        echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Mac arm64 ${COLOR_OFF}"
        wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${TORCH_VERSION}.zip
        unzip libtorch-macos-arm64-${TORCH_VERSION}.zip
        rm libtorch-macos-arm64-${TORCH_VERSION}.zip
      fi
    fi
  elif [ "$PLATFORM" = "Windows" ]; then
      echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Windows ${COLOR_OFF}"
      # TODO: Windows
      echo -e "${COLOR_RED}[ ERROR ] Unknown platform: $PLATFORM ${COLOR_OFF}"
      exit 1
  else  # Linux
    if [ -d "$DEPS_DIR/libtorch" ]; then
      RAW_VERSION=`cat "$DEPS_DIR/libtorch/build-version"`
      VERSION=`cat "$DEPS_DIR/libtorch/build-version" | cut -d "+" -f 1`
      if [ "$USE_NIGHTLIES" = "true" ] && [[ ! "${RAW_VERSION}" =~ .*"dev".* ]]; then
        rm -rf "$DEPS_DIR/libtorch"
      elif [ "$USE_NIGHTLIES" == "" ] && [ "$VERSION" != "$TORCH_VERSION" ]; then
        rm -rf "$DEPS_DIR/libtorch"
      fi
    fi
    if [ ! -d "$DEPS_DIR/libtorch" ]; then
      cd "$DEPS_DIR" || exit
      echo -e "${COLOR_GREEN}[ INFO ] Install libtorch on Linux ${COLOR_OFF}"
      if [ "$USE_NIGHTLIES" == true ]; then
        URL=https://download.pytorch.org/libtorch/nightly/${CUDA}/libtorch-cxx11-abi-shared-with-deps-latest.zip
      else
        URL=https://download.pytorch.org/libtorch/${CUDA}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${CUDA}.zip
      fi
      wget $URL
      ZIP_FILE=$(basename "$URL")
      ZIP_FILE="${ZIP_FILE//%2B/+}"
      unzip $ZIP_FILE
      rm $ZIP_FILE
    fi
    echo -e "${COLOR_GREEN}[ INFO ] libtorch is installed ${COLOR_OFF}"
  fi

  cd "$BWD" || exit
}

function install_yaml_cpp() {
  YAML_CPP_SRC_DIR=$BASE_DIR/third-party/yaml-cpp
  YAML_CPP_BUILD_DIR=$DEPS_DIR/yaml-cpp-build

  if [ ! -d "$YAML_CPP_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning yaml-cpp repo ${COLOR_OFF}"
    git clone https://github.com/jbeder/yaml-cpp.git "$YAML_CPP_SRC_DIR"
    cd $YAML_CPP_SRC_DIR
    git checkout tags/0.8.0
  fi

  if [ ! -d "$YAML_CPP_BUILD_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Building yaml-cpp ${COLOR_OFF}"

    if [ "$PLATFORM" = "Linux" ]; then
      SUDO="sudo"
    elif [ "$PLATFORM" = "Mac" ]; then
      SUDO=""
    fi

    mkdir $YAML_CPP_BUILD_DIR
    cd $YAML_CPP_BUILD_DIR
    cmake $YAML_CPP_SRC_DIR -DYAML_BUILD_SHARED_LIBS=ON -DYAML_CPP_BUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="-fPIC"
    $SUDO make install

    echo -e "${COLOR_GREEN}[ INFO ] yaml-cpp is installed ${COLOR_OFF}"
  fi

  cd "$BWD" || exit
}

function build_llama_cpp() {
  BWD=$(pwd)
  LLAMA_CPP_SRC_DIR=$BASE_DIR/third-party/llama.cpp
  cd "${LLAMA_CPP_SRC_DIR}"
  if [ "$PLATFORM" = "Mac" ]; then
    make LLAMA_METAL=OFF -j
  else
    make -j
  fi
  cd "$BWD" || exit
}

function prepare_test_files() {
  echo -e "${COLOR_GREEN}[ INFO ]Preparing test files ${COLOR_OFF}"
  local EX_DIR="${TR_DIR}/examples/"
  rsync -a --link-dest=../../test/resources/ ${BASE_DIR}/test/resources/ ${TR_DIR}/
  if [ ! -f "${EX_DIR}/babyllama/babyllama_handler/tokenizer.bin" ]; then
    wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -O "${EX_DIR}/babyllama/babyllama_handler/tokenizer.bin"
  fi
  if [ ! -f "${EX_DIR}/babyllama/babyllama_handler/stories15M.bin" ]; then
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O "${EX_DIR}/babyllama/babyllama_handler/stories15M.bin"
  fi
  # PT2.2 torch.expport does not support Mac
  if [ "$PLATFORM" = "Linux" ]; then
    if [ ! -f "${EX_DIR}/aot_inductor/llama_handler/stories15M.so" ]; then
      local HANDLER_DIR=${EX_DIR}/aot_inductor/llama_handler/
      if [ ! -f "${HANDLER_DIR}/stories15M.pt" ]; then
        wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt?download=true -O "${HANDLER_DIR}/stories15M.pt"
      fi
      local LLAMA_SO_DIR=${BASE_DIR}/third-party/llama2.so/
      PYTHONPATH=${LLAMA_SO_DIR}:${PYTHONPATH} python ${BASE_DIR}/../examples/cpp/aot_inductor/llama2/compile.py --checkpoint ${HANDLER_DIR}/stories15M.pt ${HANDLER_DIR}/stories15M.so
    fi
    if [ ! -f "${EX_DIR}/aot_inductor/resnet_handler/resne50_pt2.so" ]; then
      local HANDLER_DIR=${EX_DIR}/aot_inductor/resnet_handler/
      cd ${HANDLER_DIR}
      python ${BASE_DIR}/../examples/cpp/aot_inductor/resnet/resnet50_torch_export.py
    fi
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

  MAYBE_NIGHTLIES="-Dnightlies=OFF"
  if [ "$USE_NIGHTLIES" == true ]; then
    MAYBE_NIGHTLIES="-Dnightlies=ON"
  fi

  # Build torchserve_cpp with cmake
  cd "$BWD" || exit
  YAML_CPP_CMAKE_DIR=$DEPS_DIR/yaml-cpp-build
  FOLLY_CMAKE_DIR=$DEPS_DIR/folly-build/installed
  find $FOLLY_CMAKE_DIR -name "lib*.*"  -exec ln -s "{}" $LIBS_DIR/ \;
  if [ "$PLATFORM" = "Linux" ]; then
    cmake                                                                                     \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR;$FOLLY_CMAKE_DIR;$YAML_CPP_CMAKE_DIR;$DEPS_DIR/libtorch"                       \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"                                                          \
    "$MAYBE_BUILD_QUIC"                                                                       \
    "$MAYBE_BUILD_TESTS"                                                                      \
    "$MAYBE_BUILD_SHARED_LIBS"                                                                \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                                                               \
    "$MAYBE_USE_STATIC_DEPS"                                                                  \
    "$MAYBE_LIB_FUZZING_ENGINE"                                                               \
    "$MAYBE_CUDA_COMPILER"                                                                    \
    "$MAYBE_NIGHTLIES"                                                                        \
    ..

    if [ "$CUDA" = "cu118" ] || [ "$CUDA" = "cu121" ]; then
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/bin/nvcc
    fi
  elif [ "$PLATFORM" = "Mac" ]; then
    cmake                                                                                     \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR;$FOLLY_CMAKE_DIR;$YAML_CPP_CMAKE_DIR;$DEPS_DIR/libtorch"    \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"                                                          \
    "$MAYBE_BUILD_QUIC"                                                                       \
    "$MAYBE_BUILD_TESTS"                                                                      \
    "$MAYBE_BUILD_SHARED_LIBS"                                                                \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                                                               \
    "$MAYBE_USE_STATIC_DEPS"                                                                  \
    "$MAYBE_LIB_FUZZING_ENGINE"                                                               \
    "$MAYBE_NIGHTLIES"                                                                        \
    ..

    export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/opt/icu4c/lib
  else
    # TODO: Windows
    echo -e "${COLOR_RED}[ ERROR ] Unknown platform: $PLATFORM ${COLOR_OFF}"
    exit 1
  fi

  make -j "$JOBS"
  make format
  make install
  echo -e "${COLOR_GREEN}torchserve_cpp build is complete. To run unit test: \
  ./_build/test/torchserve_cpp_test ${COLOR_OFF}"

  cd $DEPS_DIR/../..
  if [ -f "$DEPS_DIR/../test/torchserve_cpp_test" ]; then
    $DEPS_DIR/../test/torchserve_cpp_test
  else
    echo -e "${COLOR_RED}[ ERROR ] _build/test/torchserve_cpp_test not exist ${COLOR_OFF}"
    exit 1
  fi
}

function symlink_torch_libs() {
  if [ "$PLATFORM" = "Linux" ]; then
    ln -sf ${DEPS_DIR}/libtorch/lib/*.so* ${LIBS_DIR}
  fi
}

function symlink_yaml_cpp_lib() {
  if [ "$PLATFORM" = "Linux" ]; then
    ln -sf ${DEPS_DIR}/yaml-cpp-build/*.so* ${LIBS_DIR}
  elif [ "$PLATFORM" = "Mac" ]; then
    ln -sf ${DEPS_DIR}/yaml-cpp-build/*.dylib* ${LIBS_DIR}
  fi
}

function install_torchserve_cpp() {
  TARGET_DIR=$BASE_DIR/../ts/cpp/

  if [ -d $TARGET_DIR ]; then
    rm -rf $TARGET_DIR
  fi
  mkdir $TARGET_DIR
  cp -rp $BASE_DIR/_build/bin $TARGET_DIR/bin
  cp -rp $BASE_DIR/_build/libs $TARGET_DIR/lib
  cp -rp $BASE_DIR/_build/resources $TARGET_DIR/resources
}

# Parse args
JOBS=8
WITH_QUIC=false
INSTALL_DEPENDENCIES=false
PREFIX=""
COMPILER_FLAGS=""
CUDA="cpu"
USAGE="./build.sh [-j num_jobs] [-g cu118|cu121] [-q|--with-quic] [-t|--no-tets] [-p|--prefix] [-x|--compiler-flags] [-n|--nighlies]"
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
    -n | --nightlies )
                  USE_NIGHTLIES=true
      ;;
    * )           echo $USAGE
                  exit 1
esac
shift
done

detect_platform

MAYBE_OVERRIDE_CXX_FLAGS=""
if [ -n "$COMPILER_FLAGS" ] ; then
  MAYBE_OVERRIDE_CXX_FLAGS="-DCMAKE_CXX_FLAGS=$COMPILER_FLAGS"
fi

BUILD_DIR=$BASE_DIR/_build
if [ ! -d "$BUILD_DIR" ] ; then
  mkdir -p $BUILD_DIR
fi

set -e nounset
trap 'cd $BASE_DIR' EXIT
cd $BUILD_DIR || exit
BWD=$(pwd)
DEPS_DIR=$BWD/_deps
LIBS_DIR=$BWD/libs
TR_DIR=$BWD/test/resources/
mkdir -p "$DEPS_DIR"
mkdir -p "$LIBS_DIR"
mkdir -p "$TR_DIR"

# Must execute from the directory containing this script
cd $BASE_DIR

git submodule update --init --recursive

install_folly
install_kineto
install_libtorch
install_yaml_cpp
build_llama_cpp
prepare_test_files
build
symlink_torch_libs
symlink_yaml_cpp_lib
install_torchserve_cpp
