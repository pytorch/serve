
FetchContent_Declare(
  stories15M_pt
  URL      https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt?download=true
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/
)

FetchContent_MakeAvailable(stories15M_pt)


add_custom_command(
    OUTPUT stories15M.so
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpp/third-party/llama2.so/ python ${CMAKE_CURRENT_SOURCE_DIR}/compile.py --checkpoint ${CMAKE_CURRENT_BINARY_DIR}/\'stories15M.pt?download=true\' ${CMAKE_CURRENT_BINARY_DIR}/stories15M.so
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/compile.py
)


add_library(llama2_so STATIC ../../../../cpp/third-party/llama2.so/run.cpp)
target_compile_options(llama2_so PRIVATE -Wall -Wextra -Ofast -fpermissive)

add_library(llama_so_handler SHARED src/llama_handler.cc stories15M.so)
target_link_libraries(llama_so_handler PRIVATE llama2_so ts_backends_core ts_utils ${TORCH_LIBRARIES})
