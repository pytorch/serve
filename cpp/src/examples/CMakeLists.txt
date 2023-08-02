set(MNIST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/mnist")

set(MNIST_SOURCE_FILES "")
list(APPEND MNIST_SOURCE_FILES ${MNIST_SRC_DIR}/mnist_handler.cc)
add_library(mnist_handler SHARED ${MNIST_SOURCE_FILES})
target_include_directories(mnist_handler PUBLIC ${MNIST_SRC_DIR})
target_link_libraries(mnist_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 


set(RESNET_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/resnet-18")

set(RESNET_SOURCE_FILES "")
list(APPEND RESNET_SOURCE_FILES ${RESNET_SRC_DIR}/resnet-18_handler.cc)
add_library(resnet-18_handler SHARED ${RESNET_SOURCE_FILES})
target_include_directories(resnet-18_handler PUBLIC ${RESNET_SRC_DIR})
target_link_libraries(resnet-18_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 
