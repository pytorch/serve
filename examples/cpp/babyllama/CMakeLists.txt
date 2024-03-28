
FetchContent_Declare(
  llama2_c
  GIT_REPOSITORY https://github.com/karpathy/llama2.c
  GIT_TAG d986206
)

FetchContent_Declare(
    stories15M_bin
  URL      https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/
)

FetchContent_Declare(
  tokenizer_bin
  URL      https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/
)

FetchContent_MakeAvailable(tokenizer_bin stories15M_bin llama2_c)

add_library(llama2_c STATIC ${llama2_c_SOURCE_DIR}/run.c)
target_compile_options(llama2_c PRIVATE -Wall -Wextra -Ofast -fPIC)

add_library(babyllama_handler SHARED src/baby_llama_handler.cc)
target_link_libraries(babyllama_handler PRIVATE llama2_c ts_backends_core ts_utils ${TORCH_LIBRARIES})
