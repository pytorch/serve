add_library(resnet_handler SHARED src/resnet_handler.cc)
target_link_libraries(resnet_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
