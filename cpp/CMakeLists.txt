cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
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

  add_custom_target(format
    COMMAND
      ${CLANG_FORMAT_EXE} -i -style=google ${ALL_CXX_SOURCE_FILES}
  )
endif()


find_package(Boost REQUIRED)
find_package(folly REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED NO_CMAKE_PATH)
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
