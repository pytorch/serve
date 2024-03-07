cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(torchserve_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -fPIC")

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
find_package(Torch REQUIRED)

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
