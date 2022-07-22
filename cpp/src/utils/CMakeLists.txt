set(TS_UTILS_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/utils")

list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/config.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/file_system.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/model_archive.cc)
list(APPEND TS_UTILS_SOURCE_FILES ${TS_UTILS_SRC_DIR}/dl_loader.cc)
add_library(ts_utils ${TS_UTILS_SOURCE_FILES})
target_include_directories(ts_utils PUBLIC ${TS_UTILS_INCLUDE_DIR})
target_link_libraries(ts_utils ${FOLLY_LIBRARIES})

add_library(foo SHARED ${TS_UTILS_SRC_DIR}/foo.cc)
target_link_libraries(foo PRIVATE ${FOLLY_LIBRARIES}) 
