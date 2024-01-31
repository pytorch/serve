set(LLAMACPP_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/third-party/llama.cpp")

add_library(llamacpp_handler SHARED src/llamacpp_handler.cc)

set(MY_OBJECT_FILES
    ${LLAMACPP_SRC_DIR}/ggml.o
    ${LLAMACPP_SRC_DIR}/llama.o
    ${LLAMACPP_SRC_DIR}/common.o
    ${LLAMACPP_SRC_DIR}/ggml-quants.o
    ${LLAMACPP_SRC_DIR}/ggml-alloc.o
    ${LLAMACPP_SRC_DIR}/grammar-parser.o
    ${LLAMACPP_SRC_DIR}/console.o
    ${LLAMACPP_SRC_DIR}/build-info.o
    ${LLAMACPP_SRC_DIR}/ggml-backend.o

)

target_sources(llamacpp_handler PRIVATE ${MY_OBJECT_FILES})
target_include_directories(llamacpp_handler PUBLIC ${LLAMACPP_SRC_DIR})
target_link_libraries(llamacpp_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
