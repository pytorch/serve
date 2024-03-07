cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(torchserve_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -fPIC -D_GLIBCXX_USE_CXX11_ABI=1")

find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")

find_program(CLANG_FORMAT_EXE NAMES "clang-format" REQUIRED)
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.hh
    ${PROJECT_SOURCE_DIR}/test/*.cc
    ${PROJECT_SOURCE_DIR}/test/*.hh
  )

endif()


find_package(Boost REQUIRED)
find_package(folly REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(Torch REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${FOLLY_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${GMOCK_INCLUDE_DIRS})

include_directories("${CMAKE_INSTALL_PREFIX}/_deps")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(FOLLY_LIBRARIES Folly::folly)

# add subdirectories in src
add_subdirectory(src/utils)
add_subdirectory(src/backends)
add_subdirectory(src/examples)
add_subdirectory(test)

FILE(COPY src/resources/logging.config DESTINATION "${CMAKE_INSTALL_PREFIX}/resources")
