set(MNIST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/image_classifier/mnist")

set(MNIST_SOURCE_FILES "")
list(APPEND MNIST_SOURCE_FILES ${MNIST_SRC_DIR}/mnist_handler.cc)
add_library(mnist_handler SHARED ${MNIST_SOURCE_FILES})
target_include_directories(mnist_handler PUBLIC ${MNIST_SRC_DIR})
target_link_libraries(mnist_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES}) 


set(BERT_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/src/examples/text_classifier/bert")
set(BERT_SOURCE_FILES "")
list(APPEND BERT_SOURCE_FILES ${BERT_SRC_DIR}/bert_handler.cc)
add_library(bert_handler SHARED ${BERT_SOURCE_FILES})
target_link_libraries(bert_handler PRIVATE /home/ubuntu/serve/cpp/src/examples/text_classifier/bert/libtokenizers.so)
target_include_directories(bert_handler PUBLIC ${BERT_SRC_DIR})
target_link_libraries(bert_handler PRIVATE ts_backends_torch_scripted ts_utils ${TORCH_LIBRARIES})
