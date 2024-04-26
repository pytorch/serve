add_subdirectory("../../../examples/cpp/babyllama/" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/babyllama/babyllama_handler/")

add_subdirectory("../../../examples/cpp/llamacpp/" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/llamacpp/llamacpp_handler/")

add_subdirectory("../../../examples/cpp/mnist/" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/mnist/mnist_handler/")

# PT2.2 torch.expport does not support Mac
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_subdirectory("../../../examples/cpp/aot_inductor/llama2/" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/aot_inductor/llama_handler/")

  add_subdirectory("../../../examples/cpp/aot_inductor/bert" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/aot_inductor/bert_handler/")

  add_subdirectory("../../../examples/cpp/aot_inductor/resnet" "${CMAKE_CURRENT_BINARY_DIR}/../../test/resources/examples/aot_inductor/resnet_handler/")
endif()
