add_library(mnist_handler SHARED src/mnist_handler.cc)

target_link_libraries(mnist_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
