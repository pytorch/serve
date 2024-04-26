
add_custom_command(
    OUTPUT resnet50_pt2.so
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/resnet50_torch_export.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/resnet50_torch_export.py
)

add_library(resnet_handler SHARED src/resnet_handler.cc resnet50_pt2.so)
target_link_libraries(resnet_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
