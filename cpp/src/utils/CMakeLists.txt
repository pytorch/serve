set(TS_UTILS_SOURCE_FILES "${torchserve_cpp_SOURCE_DIR}/src/utils")
set(TS_UTILS_INCLUDE_DIR "${torchserve_cpp_SOURCE_DIR}/src/utils")

list(APPEND TS_UTILS_SOURCE_FILES config.cc file_system.cc model_archive.cc)
add_library(ts_utils ${TS_UTILS_SOURCE_FILES})
target_include_directories(ts_utils PUBLIC ${TS_UTILS_INCLUDE_DIR})
target_link_libraries(ts_utils ${FOLLY_LIBRARIES})