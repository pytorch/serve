
add_library(babyllama_handler SHARED src/baby_llama_handler.cc)

target_link_libraries(babyllama_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES})
target_compile_options(babyllama_handler PRIVATE -Wall -Wextra -Ofast)
