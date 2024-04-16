set(LLAMACPP_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/third-party/llama.cpp")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(llamacpp_handler SHARED src/llamacpp_handler.cc)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(LLAMA_METAL OFF)
endif()

FetchContent_Declare(
  llama.cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
  GIT_TAG b2241
)
FetchContent_GetProperties(llama.cpp)

if(NOT llama.cpp_POPULATED)
  message(STATUS "Fetching llama.cpp...")
  FetchContent_Populate(llama.cpp)
  add_subdirectory(${llama.cpp_SOURCE_DIR} ${llama.cpp_BINARY_DIR})
endif()

target_include_directories(llamacpp_handler PUBLIC ${LLAMACPP_SRC_DIR})
target_link_libraries(llamacpp_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES} common llama)
