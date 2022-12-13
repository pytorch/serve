set(MNIST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/mnist")

set(MNIST_SOURCE_FILES "")
list(APPEND MNIST_SOURCE_FILES ${MNIST_SRC_DIR}/mnist_handler.cc)
add_library(mnist_handler SHARED ${MNIST_SOURCE_FILES})
target_include_directories(mnist_handler PUBLIC ${MNIST_SRC_DIR})
target_link_libraries(mnist_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 


set(TEXT_CLASSIFIER_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/text_classifier")

set(TEXT_CLASSIFIER_SOURCE_FILES "")
list(APPEND TEXT_CLASSIFIER_SOURCE_FILES ${TEXT_CLASSIFIER_SRC_DIR}/text_classifier_handler.cc)
add_library(text_classifier_handler SHARED ${TEXT_CLASSIFIER_SOURCE_FILES})
target_include_directories(text_classifier_handler PUBLIC ${TEXT_CLASSIFIER_SRC_DIR})
target_link_libraries(text_classifier_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 
