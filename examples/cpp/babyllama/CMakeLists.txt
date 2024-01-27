
add_library(llama2_c STATIC ../../../cpp/third-party/llama2.c/run.c)
target_compile_options(llama2_c PRIVATE -Wall -Wextra -Ofast -fPIC)

add_library(babyllama_handler SHARED src/baby_llama_handler.cc)
target_link_libraries(babyllama_handler PRIVATE llama2_c ts_backends_core ts_utils ${TORCH_LIBRARIES})
