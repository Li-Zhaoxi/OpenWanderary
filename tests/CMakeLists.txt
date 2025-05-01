find_package(GTest REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/add_test.cmake")


enable_testing()

file(CREATE_LINK ${CMAKE_SOURCE_DIR}/tests/test_data ${CMAKE_BINARY_DIR}/tests/test_data SYMBOLIC)

add_custom_target(tests_cpp "${CMAKE_CTEST_COMMAND}" -V -T Test VERBATIM)

add_custom_target(tests DEPENDS tests_cpp)

add_subdirectory(wanderary)
