add_library(llama2_so STATIC ../../../../cpp/third-party/llama2.so/run.cpp)
target_compile_options(llama2_so PRIVATE -Wall -Wextra -Ofast -fpermissive)

add_library(llama_so_handler SHARED src/llama_handler.cc)
target_link_libraries(llama_so_handler PRIVATE llama2_so ts_backends_core ts_utils ${TORCH_LIBRARIES})
