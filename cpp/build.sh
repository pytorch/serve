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
    git \
    cmake \
    m4 \
    g++ \
    flex \
    bison \
    libgflags-dev \
    libgoogle-glog-dev \
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
    libdouble-conversion-dev
}

function install_dependencies_mac() {
  # install the default dependencies from homebrew
  brew install -f            \
    cmake                    \
    m4                       \
    boost                    \
    double-conversion        \
    gflags                   \
    glog                     \
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
    glog                    \
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

function synch_dependency_to_commit() {
  # Utility function to synch a dependency to a specific commit. Takes two arguments:
  #   - $1: folder of the dependency's git repository
  #   - $2: path to the text file containing the desired commit hash
  if [ "$FETCH_DEPENDENCIES" = false ] ; then
    return
  fi
  DEP_REV=$(sed 's/Subproject commit //' "$2")
  pushd "$1"
  git fetch
  # Disable git warning about detached head when checking out a specific commit.
  git -c advice.detachedHead=false checkout "$DEP_REV"
  popd
}

function setup_fmt_cmake() {
  FMT_SRC_DIR=$BASE_DIR/third-party/fmt
  FMT_BUILD_DIR=$BASE_DIR/third-party/fmt/build
  if [ ! -d "$FMT_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning fmt repo ${COLOR_OFF}"
    git submodule add https://github.com/fmtlib/fmt.git  "$FMT_SRC_DIR"
    cd $FMT_SRC_DIR
    git fetch --tags
    git checkout 6.2.1
  fi

  echo -e "${COLOR_GREEN}Building fmt ${COLOR_OFF}"
  mkdir -p "$FMT_BUILD_DIR"
  cd "$FMT_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                   \
    -DFMT_DOC=OFF                                 \
    -DFMT_TEST=OFF                                \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}fmt is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

function setup_googletest_cmake() {
  GTEST_SRC_DIR=$BASE_DIR/third-party/googletest
  GTEST_BUILD_DIR=$BASE_DIR/third-party/googletest/build/

  if [ ! -d "$GTEST_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning googletest repo ${COLOR_OFF}"
    git submodule add https://github.com/google/googletest.git  "$GTEST_SRC_DIR"
    cd $GTEST_SRC_DIR
    git fetch --tags
    git checkout release-1.8.0
  fi
  
  echo -e "${COLOR_GREEN}Building googletest ${COLOR_OFF}"
  mkdir -p "$GTEST_BUILD_DIR"
  cd "$GTEST_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}googletest is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

function setup_folly_cmake() {
  FOLLY_SRC_DIR=$BASE_DIR/third-party/folly
  FOLLY_BUILD_DIR=$BASE_DIR/third-party/folly/build/

  if [ ! -d "$FOLLY_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning folly repo ${COLOR_OFF}"
    git submodule add https://github.com/facebook/folly.git "$FOLLY_SRC_DIR"
    cd $FOLLY_SRC_DIR
    git checkout tags/v2021.10.18.00
    #synch_dependency_to_commit "$FOLLY_DIR" "$BASE_DIR"/../build/deps/github_hashes/facebook/foll
  fi
  
  if [ "$PLATFORM" = "Mac" ]; then
    # Homebrew installs OpenSSL in a non-default location on MacOS >= Mojave
    # 10.14 because MacOS has its own SSL implementation.  If we find the
    # typical Homebrew OpenSSL dir, load OPENSSL_ROOT_DIR so that cmake
    # will find the Homebrew version.
    dir=/usr/local/opt/openssl
    if [ -d $dir ]; then
        export OPENSSL_ROOT_DIR=$dir
    fi
  fi
  echo -e "${COLOR_GREEN}Building Folly ${COLOR_OFF}"
  mkdir -p "$FOLLY_BUILD_DIR"
  cd "$FOLLY_BUILD_DIR" || exit
  MAYBE_DISABLE_JEMALLOC=""
  if [ "$NO_JEMALLOC" == true ] ; then
    MAYBE_DISABLE_JEMALLOC="-DFOLLY_USE_JEMALLOC=0"
  fi

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    -DBUILD_TESTS=OFF                             \
    "$MAYBE_USE_STATIC_DEPS"                      \
    "$MAYBE_USE_STATIC_BOOST"                     \
    "$MAYBE_BUILD_SHARED_LIBS"                    \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                   \
    $MAYBE_DISABLE_JEMALLOC                       \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}Folly is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

setup_spdlog_cmake() {
  SPDLOG_SRC_DIR=$BASE_DIR/third-party/spdlog
  SPDLOG_BUILD_DIR=$DEPS_DIR/spdlog

  if [ ! -d "$SPDLOG_SRC_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning spdlog repo ${COLOR_OFF}"
    git submodule add https://github.com/gabime/spdlog.git  "$SPDLOG_SRC_DIR"
    cd "$SPDLOG_SRC_DIR"
    git fetch --tags
    git checkout v1.9.2
  fi

  cd "$BASE_DIR"
  # https://spdlog.docsforge.com/v1.x/9.cmake/#generate
  echo -e "${COLOR_GREEN}Installing spdlog ${COLOR_OFF}"

  # install
  cmake                                           \
    -S $SPDLOG_SRC_DIR                            \
    -B $SPDLOG_BUILD_DIR                          \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
  
  cmake --build "$SPDLOG_BUILD_DIR" --target install       

  echo -e "${COLOR_GREEN}spdlog is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

setup_concurrentqueue_cmake() {
  CONCURRENTQUEUE_SRC_DIR=$BASE_DIR/third-party/concurrentqueue
  CONCURRENTQUEUE_BUILD_DIR=$DEPS_DIR/concurrentqueue

  if [ ! -d "$CONCURRENTQUEUE_SRC_DIR" ] ; then
    cd "$BASE_DIR/third-party"
    echo -e "${COLOR_GREEN}[ INFO ] Cloning concurrentqueue repo ${COLOR_OFF}"
    git submodule add https://github.com/cameron314/concurrentqueue.git "$CONCURRENTQUEUE_SRC_DIR"
    cd "$CONCURRENTQUEUE_SRC_DIR"
    git fetch --tags
    git checkout v1.0.3
  fi

  cd "$BASE_DIR"
  echo -e "${COLOR_GREEN}Installing concurrentqueue ${COLOR_OFF}"

  # install
  cmake                                           \
    -S $CONCURRENTQUEUE_SRC_DIR                   \
    -B $CONCURRENTQUEUE_BUILD_DIR                 \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
  
  cmake --build "$CONCURRENTQUEUE_BUILD_DIR" --target install       

  echo -e "${COLOR_GREEN}concurrentqueue is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

function setup_fmt() {
  FMT_DIR=$DEPS_DIR/fmt
  FMT_BUILD_DIR=$DEPS_DIR/fmt/build/

  if [ ! -d "$FMT_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning fmt repo ${COLOR_OFF}"
    git clone https://github.com/fmtlib/fmt.git  "$FMT_DIR"
  fi
  cd "$FMT_DIR"
  git fetch --tags
  git checkout 6.2.1
  echo -e "${COLOR_GREEN}Building fmt ${COLOR_OFF}"
  mkdir -p "$FMT_BUILD_DIR"
  cd "$FMT_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                   \
    -DFMT_DOC=OFF                                 \
    -DFMT_TEST=OFF                                \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}fmt is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

function setup_googletest() {
  GTEST_DIR=$DEPS_DIR/googletest
  GTEST_BUILD_DIR=$DEPS_DIR/googletest/build/

  if [ ! -d "$GTEST_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning googletest repo ${COLOR_OFF}"
    git clone https://github.com/google/googletest.git  "$GTEST_DIR"
  fi
  cd "$GTEST_DIR"
  git fetch --tags
  git checkout release-1.8.0
  echo -e "${COLOR_GREEN}Building googletest ${COLOR_OFF}"
  mkdir -p "$GTEST_BUILD_DIR"
  cd "$GTEST_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}googletest is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

function setup_folly() {
  FOLLY_DIR=$DEPS_DIR/folly
  FOLLY_BUILD_DIR=$DEPS_DIR/folly/build/

  if [ ! -d "$FOLLY_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning folly repo ${COLOR_OFF}"
    git clone https://github.com/facebook/folly.git "$FOLLY_DIR"
  fi
  synch_dependency_to_commit "$FOLLY_DIR" "$BASE_DIR"/../build/deps/github_hashes/facebook/folly-rev.txt
  if [ "$PLATFORM" = "Mac" ]; then
    # Homebrew installs OpenSSL in a non-default location on MacOS >= Mojave
    # 10.14 because MacOS has its own SSL implementation.  If we find the
    # typical Homebrew OpenSSL dir, load OPENSSL_ROOT_DIR so that cmake
    # will find the Homebrew version.
    dir=/usr/local/opt/openssl
    if [ -d $dir ]; then
        export OPENSSL_ROOT_DIR=$dir
    fi
  fi
  echo -e "${COLOR_GREEN}Building Folly ${COLOR_OFF}"
  mkdir -p "$FOLLY_BUILD_DIR"
  cd "$FOLLY_BUILD_DIR" || exit
  MAYBE_DISABLE_JEMALLOC=""
  if [ "$NO_JEMALLOC" == true ] ; then
    MAYBE_DISABLE_JEMALLOC="-DFOLLY_USE_JEMALLOC=0"
  fi

  MAYBE_USE_STATIC_DEPS=""
  MAYBE_USE_STATIC_BOOST=""
  MAYBE_BUILD_SHARED_LIBS=""
  if [ "$BUILD_FOR_FUZZING" == true ] ; then
    MAYBE_USE_STATIC_DEPS="-DUSE_STATIC_DEPS_ON_UNIX=ON"
    MAYBE_USE_STATIC_BOOST="-DBOOST_LINK_STATIC=ON"
    MAYBE_BUILD_SHARED_LIBS="-DBUILD_SHARED_LIBS=OFF"
  fi

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    -DBUILD_TESTS=OFF                             \
    "$MAYBE_USE_STATIC_DEPS"                      \
    "$MAYBE_USE_STATIC_BOOST"                     \
    "$MAYBE_BUILD_SHARED_LIBS"                    \
    "$MAYBE_OVERRIDE_CXX_FLAGS"                   \
    $MAYBE_DISABLE_JEMALLOC                       \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}Folly is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

setup_spdlog() {
  SPDLOG_DIR=$DEPS_DIR/spdlog
  SPDLOG_BUILD_DIR=$DEPS_DIR/spdlog/build/

  if [ ! -d "$SPDLOG_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning spdlog repo ${COLOR_OFF}"
    git clone https://github.com/gabime/spdlog.git "$SPDLOG_DIR"
  fi
  cd "$SPDLOG_DIR"
  git fetch --tags
  git checkout v1.9.2
  echo -e "${COLOR_GREEN}Building spdlog ${COLOR_OFF}"
  mkdir -p "$SPDLOG_BUILD_DIR"
  cd "$SPDLOG_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}spdlog is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

setup_concurrentqueue() {
  CONCURRENTQUEUE_DIR=$DEPS_DIR/concurrentqueue
  CONCURRENTQUEUE_BUILD_DIR=$DEPS_DIR/concurrentqueue/build/

  if [ ! -d "$CONCURRENTQUEUE_DIR" ] ; then
    echo -e "${COLOR_GREEN}[ INFO ] Cloning concurrentqueue repo ${COLOR_OFF}"
    git clone https://github.com/cameron314/concurrentqueue.git "$CONCURRENTQUEUE_DIR"
  fi
  cd "$CONCURRENTQUEUE_DIR"
  git fetch --tags
  git checkout v1.0.3
  echo -e "${COLOR_GREEN}Building concurrentqueue ${COLOR_OFF}"
  mkdir -p "$CONCURRENTQUEUE_BUILD_DIR"
  cd "$CONCURRENTQUEUE_BUILD_DIR" || exit

  cmake                                           \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR"               \
    -DCMAKE_INSTALL_PREFIX="$DEPS_DIR"            \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo             \
    ..
  make -j "$JOBS"
  make install
  echo -e "${COLOR_GREEN}concurrentqueue is installed ${COLOR_OFF}"
  cd "$BWD" || exit
}

# Parse args
JOBS=8
WITH_QUIC=false
INSTALL_DEPENDENCIES=false
FETCH_DEPENDENCIES=true
PREFIX=""
COMPILER_FLAGS=""
USAGE="./build.sh [-j num_jobs] [-q|--with-quic] [-m|--no-jemalloc] [--no-install-dependencies] [-p|--prefix] [-x|--compiler-flags] [--no-fetch-dependencies]"
while [ "$1" != "" ]; do
  case $1 in
    -j | --jobs ) shift
                  JOBS=$1
                  ;;
    -q | --with-quic )
                  WITH_QUIC=true
                  ;;
    -m | --no-jemalloc )
                  NO_JEMALLOC=true
                  ;;
    --no-install-dependencies )
                  INSTALL_DEPENDENCIES=false
          ;;
    --no-fetch-dependencies )
                  FETCH_DEPENDENCIES=false
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
DEPS_DIR=$BWD/deps
mkdir -p "$DEPS_DIR"

# Must execute from the directory containing this script
cd "$(dirname "$0")"

setup_fmt_cmake
#setup_fmt
#setup_googletest
#setup_folly
setup_folly_cmake
#setup_spdlog
setup_spdlog_cmake
#setup_concurrentqueue
setup_concurrentqueue_cmake

rest() {
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

# Build ModelServerCPP with cmake
cd "$BWD" || exit
cmake                                     \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo       \
  -DCMAKE_PREFIX_PATH="$DEPS_DIR"         \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"        \
  "$MAYBE_BUILD_QUIC"                     \
  "$MAYBE_BUILD_TESTS"                    \
  "$MAYBE_BUILD_SHARED_LIBS"              \
  "$MAYBE_OVERRIDE_CXX_FLAGS"             \
  "$MAYBE_USE_STATIC_DEPS"                \
  "$MAYBE_LIB_FUZZING_ENGINE"             \
  ../..

make -j "$JOBS"
echo -e "${COLOR_GREEN}Proxygen build is complete. To run unit test: \
  cd _build/ && make test ${COLOR_OFF}"
}