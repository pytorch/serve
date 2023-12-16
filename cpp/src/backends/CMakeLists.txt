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
target_link_libraries(ts_backends_protocol PRIVATE ts_utils ${FOLLY_LIBRARIES})
install(TARGETS ts_backends_protocol DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/libs)

# build library ts_backend_core
set(TS_BACKENDS_CORE_SOURCE_FILES "")
list(APPEND TS_BACKENDS_CORE_SOURCE_FILES ${TS_BACKENDS_CORE_SRC_DIR}/backend.cc)
add_library(ts_backends_core SHARED ${TS_BACKENDS_CORE_SOURCE_FILES})
target_include_directories(ts_backends_core PUBLIC ${TS_BACKENDS_CORE_SRC_DIR})
target_link_libraries(ts_backends_core PRIVATE ts_utils ts_backends_protocol ${FOLLY_LIBRARIES})
install(TARGETS ts_backends_core DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/libs)

# build library ts_backend_torch_scripted
set(TS_BACKENDS_TORCH_SCRIPTED_SOURCE_FILES "")
list(APPEND TS_BACKENDS_TORCH_SCRIPTED_SOURCE_FILES ${TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR}/torch_scripted_backend.cc ${TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR}/model_instance.cc)
list(APPEND TS_BACKENDS_TORCH_SCRIPTED_SOURCE_FILES ${TS_BACKENDS_SRC_DIR}/handler/base_handler.cc)
add_library(ts_backends_torch_scripted SHARED ${TS_BACKENDS_TORCH_SCRIPTED_SOURCE_FILES})
target_include_directories(ts_backends_torch_scripted PUBLIC
  ${TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR} ${TS_BACKENDS_TORCH_SCRIPTED_SRC_DIR}/handler ${TORCH_INCLUDE_DIRS})
target_link_libraries(ts_backends_torch_scripted PUBLIC ts_utils ts_backends_core ${TORCH_LIBRARIES})
install(TARGETS ts_backends_torch_scripted DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/libs)

# build library ts_backend_torch_deploy
#set(TS_BACKENDS_TORCH_DEPLOY_SOURCE_FILES "")
#add_library(ts_backends_torch_deploy SHARED ${TS_BACKENDS_TORCH_DEPLOY_SOURCE_FILES})
#target_include_directories(ts_backends_torch_deploy PUBLIC ${TS_BACKENDS_TORCH_DEPLOY_SRC_DIR})
#target_link_libraries(ts_backends_torch_deploy PRIVATE ts_utils ts_backends_core ${TORCH_LIBRARIES})

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
  PRIVATE ts_backends_core ts_backends_protocol ts_backends_torch_scripted ${FOLLY_LIBRARIES})
install(TARGETS model_worker_socket DESTINATION ${torchserve_cpp_SOURCE_DIR}/_build/bin)
