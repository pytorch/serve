cmake_minimum_required(VERSION 3.10)
project(pt_c_inference)

set(CMAKE_PREFIX_PATH "/Users/demo/.pyenv/versions/3.6.9/lib/python3.6/site-packages/torch/")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(pybind11)
pybind11_add_module(pt_c_inference src/pt_c_inference.cpp)

target_link_libraries (pt_c_inference PRIVATE "${TORCH_LIBRARIES}")


