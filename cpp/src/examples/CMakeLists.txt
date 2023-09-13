set(MNIST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/mnist")

set(MNIST_SOURCE_FILES "")
list(APPEND MNIST_SOURCE_FILES ${MNIST_SRC_DIR}/mnist_handler.cc)
add_library(mnist_handler SHARED ${MNIST_SOURCE_FILES})
target_include_directories(mnist_handler PUBLIC ${MNIST_SRC_DIR})
target_link_libraries(mnist_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 

set(LLM_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/llamacpp")
set(LLAMACPP_SRC_DIR "/home/ubuntu/llama.cpp")
set(LLM_SOURCE_FILES "")
list(APPEND LLM_SOURCE_FILES ${LLM_SRC_DIR}/llamacpp_handler.cc)
add_library(llamacpp_handler SHARED ${LLM_SOURCE_FILES})
target_include_directories(llamacpp_handler PUBLIC ${LLM_SRC_DIR})
target_include_directories(llamacpp_handler PUBLIC ${LLAMACPP_SRC_DIR})
target_link_libraries(llamacpp_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES})


set(MY_OBJECT_FILES
    ${LLAMACPP_SRC_DIR}/ggml.o
    ${LLAMACPP_SRC_DIR}/llama.o
    ${LLAMACPP_SRC_DIR}/common.o
    ${LLAMACPP_SRC_DIR}/k_quants.o
    ${LLAMACPP_SRC_DIR}/ggml-alloc.o
    ${LLAMACPP_SRC_DIR}/grammar-parser.o
    ${LLAMACPP_SRC_DIR}/console.o

)

target_sources(llamacpp_handler PRIVATE ${MY_OBJECT_FILES})
