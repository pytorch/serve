cmake_minimum_required(VERSION 3.10)
project(torchserve_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# add subdirectories in src
add_subdirectory(backend)
list(APPEND TS_LIBS backend)

# add_subdirectory(src/core)
# list(APPEND TS_LIBS core)

# add_subdirectory(src/protocol)
# list(APPEND TS_LIBS protocol)

# add_subdirectory(src/socket)
# list(APPEND TS_LIBS socket)

# add_subdirectory(util)
# list(APPEND TS_LIBS util)

add_executable(model_service_worker model_service_worker.cc)
target_link_libraries(model_service_worker PUBLIC ${TS_LIBS})
target_compile_features(model_service_worker PUBLIC cxx_std_17)

# Third-party libraries
find_package(spdlog CONFIG REQUIRED)
list(APPEND TS_LIBS spdlog)

find_package(Torch REQUIRED)
target_include_directories(model_service_worker PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(model_service_worker PRIVATE ${TORCH_LIBRARIES})

find_package(glog REQUIRED)
target_link_libraries(model_service_worker PRIVATE glog)

find_package(folly CONFIG REQUIRED)
list(APPEND TS_LIBS folly)

find_package(fmt CONFIG REQUIRED)
list(APPEND TS_LIBS fmt)

include_directories("${CMAKE_INSTALL_PREFIX}/deps/include")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})