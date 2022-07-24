cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(torchserve_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra")

find_package(Boost REQUIRED)
find_package(folly REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${FOLLY_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

include_directories("${CMAKE_INSTALL_PREFIX}/_deps")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(FOLLY_LIBRARIES Folly::folly)

# add subdirectories in src
add_subdirectory(src/utils)
add_subdirectory(src/backends)
add_subdirectory(test)