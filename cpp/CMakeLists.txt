cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(torchserve_cpp VERSION 0.1)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -fPIC")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})

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

execute_process(COMMAND python -c "import importlib.util; print(importlib.util.find_spec('nvidia') is not None)"
                OUTPUT_VARIABLE NVIDIA_AVAILABLE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(${NVIDIA_AVAILABLE} STREQUAL "True")
  execute_process(COMMAND python -c "import nvidia;from pathlib import Path;print(Path(nvidia.__file__).parent/'nccl'/'lib')"
                  OUTPUT_VARIABLE NCCL_PATH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.2")
endif()

execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Boost REQUIRED)
find_package(Torch REQUIRED)

find_library(NCCL_LIBRARY nccl HINTS ${NCCL_PATH})
if("${NCCL_LIBRARY}" STREQUAL "NCCL_LIBRARY-NOTFOUND")
  set(NCCL_LIBRARY "")
endif()

include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.13.0
)
FetchContent_GetProperties(spdlog)

if(NOT spdlog_POPULATED)
  message(STATUS "Fetching spdlog...")
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()


FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.11.3
)
FetchContent_GetProperties(json)

if(NOT json_POPULATED)
  message(STATUS "Fetching json...")
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
endif()

include_directories("${json_SOURCE_DIR}/include" "${spdlog_SOURCE_DIR}/include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${GMOCK_INCLUDE_DIRS})

include_directories("${CMAKE_INSTALL_PREFIX}/_deps")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add subdirectories in src
add_subdirectory(src/utils)
add_subdirectory(src/backends)
add_subdirectory(src/examples)
add_subdirectory(test)

FILE(COPY src/resources/logging.yaml DESTINATION "${CMAKE_INSTALL_PREFIX}/resources")

install(SCRIPT "install.cmake")
