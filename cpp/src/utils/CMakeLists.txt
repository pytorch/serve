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


FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.2.1
)
FetchContent_GetProperties(fmt)

if(NOT fmt_POPULATED)
  message(STATUS "Fetching fmt...")
  FetchContent_Populate(fmt)
  add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR})
endif()


set(TS_UTILS_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/utils")

list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/config.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/file_system.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/json.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/model_archive.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/logging.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/metrics/units.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/metrics/log_metric.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/metrics/yaml_config.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/metrics/log_metrics_cache.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/metrics/registry.cc)
add_library(ts_utils SHARED ${TS_UTILS_SOURCE_FILES})
target_include_directories(ts_utils PUBLIC ${TS_UTILS_SRC_DIR})
target_include_directories(ts_utils PRIVATE ${Boost_INCLUDE_DIRS})
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  target_link_libraries(ts_utils ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} yaml-cpp::yaml-cpp spdlog::spdlog nlohmann_json::nlohmann_json fmt::fmt)
else()
  target_link_libraries(ts_utils ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} yaml-cpp nlohmann_json fmt)
endif()

install(TARGETS ts_utils DESTINATION ${CMAKE_INSTALL_PREFIX}/libs)

list(APPEND FOO_SOURCE_FILES ${TS_UTILS_SRC_DIR}/ifoo.hh)
list(APPEND FOO_SOURCE_FILES ${TS_UTILS_SRC_DIR}/foo.hh)
list(APPEND FOO_SOURCE_FILES ${TS_UTILS_SRC_DIR}/foo.cc)
add_library(foo SHARED ${FOO_SOURCE_FILES})
target_include_directories(foo PUBLIC ${TS_UTILS_SRC_DIR})
