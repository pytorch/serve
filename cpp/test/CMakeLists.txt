include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)
# run google test
enable_testing()

set(TEST_BINARY ${CMAKE_PROJECT_NAME}_test)
set(TS_TEST_SRC_DIR "${torchserve_cpp_SOURCE_DIR}/test/")
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.cc *.hh)

add_executable(${TEST_BINARY} ${TEST_SOURCES})
target_link_libraries(${TEST_BINARY} gtest_main gmock_main ts_backends_torch_scripted ts_backends_protocol ts_utils)

include(GoogleTest)
gtest_discover_tests(${TEST_BINARY})