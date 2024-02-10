set(TOKENZIER_CPP_PATH $CMAKE_INSTALL_PREFIX/../third-party/tokenizers-cpp)
add_library(bert_handler SHARED src/bert_handler.cc)
target_include_directories(bert_handler PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(bert_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES} tokenizers_cpp)
