include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)
# run google test
enable_testing()

#add libtorchtext library
add_library(torchtext SHARED IMPORTED)
set_target_properties( torchtext PROPERTIES IMPORTED_LOCATION "${torchserve_cpp_SOURCE_DIR}/lib/libtorchtext.so")
target_link_libraries(torchtext INTERFACE ${TORCH_LIBRARIES})


set(TEST_BINARY ${CMAKE_PROJECT_NAME}_test)
set(TS_TEST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/test/")
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.cc *.hh)

add_executable(${TEST_BINARY} ${TEST_SOURCES})
target_link_libraries(${TEST_BINARY} gtest_main gmock_main ts_backends_torch_scripted ts_backends_protocol ts_utils torchtext)
set_target_properties(${TEST_BINARY} PROPERTIES LINK_WHAT_YOU_USE "TRUE")

include(GoogleTest)
gtest_discover_tests(${TEST_BINARY})
