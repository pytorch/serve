cmake_minimum_required(VERSION 3.10)
project(torchserve_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# add subdirectories in src
add_subdirectory(src/backend)
list(APPEND TS_LIBS backend)

add_subdirectory(src/core)
list(APPEND TS_LIBS core)

add_subdirectory(src/protocol)
list(APPEND TS_LIBS protocol)

add_subdirectory(src/socket)
list(APPEND TS_LIBS socket)

add_subdirectory(src/util)
list(APPEND TS_LIBS util)

# Third-party libraries
find_package(spdlog CONFIG REQUIRED)
list(APPEND TS_LIBS spdlog)

find_package(folly CONFIG REQUIRED)
list(APPEND TS_LIBS folly)

find_package(fmt CONFIG REQUIRED)
list(APPEND TS_LIBS fmt)

include_directories("${CMAKE_INSTALL_PREFIX}/deps/include")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})