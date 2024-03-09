FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  GIT_TAG v2.2.2
)
FetchContent_GetProperties(spdlgflagsog)

if(NOT gflags_POPULATED)
  message(STATUS "Fetching gflags...")
  FetchContent_Populate(gflags)
  add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})
endif()


set(TS_BACKENDS_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/backends")
set(TS_BACKENDS_CORE_SRC_DIR "${TS_BACKENDS_SRC_DIR}/core")
set(TS_BACKENDS_PROTOCOL_SRC_DIR "${TS_BACKENDS_SRC_DIR}/protocol")
set(TS_BACKENDS_PROCESS_SRC_DIR "${TS_BACKENDS_SRC_DIR}/process")
set(TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR "${TS_BACKENDS_SRC_DIR}/torch_scripted")
set(TS_BACKENDS_TORCH_DEPLOY_SRC_DIR "${TS_BACKENDS_SRC_DIR}/torch_deploy")

# build library TS_BACKENDS_protocol
set(TS_BACKENDS_PROTOCOL_SOURCE_FILES "")
list(APPEND TS_BACKENDS_PROTOCOL_SOURCE_FILES ${TS_BACKENDS_PROTOCOL_SRC_DIR}/otf_message.cc)
list(APPEND TS_BACKENDS_PROTOCOL_SOURCE_FILES ${TS_BACKENDS_PROTOCOL_SRC_DIR}/socket.cc)
add_library(ts_backends_protocol SHARED ${TS_BACKENDS_PROTOCOL_SOURCE_FILES})
target_include_directories(ts_backends_protocol PUBLIC ${TS_BACKENDS_PROTOCOL_SRC_DIR})
target_link_libraries(ts_backends_protocol PRIVATE ts_utils)
install(TARGETS ts_backends_protocol DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/libs)

# build library ts_backend_core
set(BACKEND_SOURCE_FILES "")
list(APPEND BACKEND_SOURCE_FILES ${TS_BACKENDS_SRC_DIR}/core/backend.cc)
list(APPEND BACKEND_SOURCE_FILES ${TS_BACKENDS_SRC_DIR}/core/model_instance.cc)
list(APPEND BACKEND_SOURCE_FILES ${TS_BACKENDS_SRC_DIR}/handler/base_handler.cc)
list(APPEND BACKEND_SOURCE_FILES ${TS_BACKENDS_SRC_DIR}/handler/torch_scripted_handler.cc)
add_library(ts_backends_core SHARED ${BACKEND_SOURCE_FILES})
target_include_directories(ts_backends_core PUBLIC ${TS_BACKENDS_CORE_SRC_DIR})
target_link_libraries(ts_backends_core PUBLIC ts_utils ts_backends_protocol ${TORCH_LIBRARIES})
install(TARGETS ts_backends_core DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/libs)

# build exe model_worker_socket
add_executable(model_worker_socket
  "${TS_BACKENDS_PROCESS_SRC_DIR}/model_worker_socket.cc"
  "${TS_BACKENDS_PROCESS_SRC_DIR}/model_worker.cc"
)
target_include_directories(model_worker_socket PRIVATE
  ${TS_BACKENDS_CORE_SRC_DIR}
  ${TS_BACKENDS_PROTOCOL_SRC_DIR}
  ${TS_BACKENDS_PROCESS_SRC_DIR}
  ${TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR}
)
target_link_libraries(model_worker_socket
  PRIVATE ts_backends_core ts_backends_protocol ${TORCH_LIBRARIES} gflags)
install(TARGETS model_worker_socket DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/bin)
