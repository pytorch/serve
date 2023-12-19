set(MNIST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/mnist")

set(MNIST_SOURCE_FILES "")
list(APPEND MNIST_SOURCE_FILES ${MNIST_SRC_DIR}/mnist_handler.cc)
add_library(mnist_handler SHARED ${MNIST_SOURCE_FILES})
target_include_directories(mnist_handler PUBLIC ${MNIST_SRC_DIR})
target_link_libraries(mnist_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})


set(BABYLLAMA_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/babyllama")
set(BABYLLAMA_SOURCE_FILES "")
list(APPEND BABYLLAMA_SOURCE_FILES ${BABYLLAMA_SRC_DIR}/baby_llama_handler.cc)
add_library(babyllama_handler SHARED ${BABYLLAMA_SOURCE_FILES})
target_include_directories(babyllama_handler PUBLIC ${BABYLLAMA_SRC_DIR})
target_link_libraries(babyllama_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
target_compile_options(babyllama_handler PRIVATE -Wall -Wextra -Ofast)
