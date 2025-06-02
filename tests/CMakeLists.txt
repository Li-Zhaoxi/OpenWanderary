

include("${CMAKE_SOURCE_DIR}/cmake/add_test.cmake")


enable_testing()

file(CREATE_LINK ${CMAKE_SOURCE_DIR}/tests/test_data ${CMAKE_BINARY_DIR}/tests/test_data SYMBOLIC)

add_custom_target(tests_cpp "${CMAKE_CTEST_COMMAND}" -V -T Test VERBATIM)

# 配置py测试
file(GLOB_RECURSE PYBIND11_PYTEST_FILES "python/[^_]*.py")
set(SYMLINK_PYTEST_FILES)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/python)
foreach(PYBIND11_PYTEST_FILE ${PYBIND11_PYTEST_FILES})
    get_filename_component(PYBIND11_PYTEST_FILE_NAME ${PYBIND11_PYTEST_FILE} NAME)

    # 创建软连接
    set(SYMLINK_PYTEST_FILE ${CMAKE_BINARY_DIR}/tests/python/${PYBIND11_PYTEST_FILE_NAME})
    file(CREATE_LINK ${PYBIND11_PYTEST_FILE} ${SYMLINK_PYTEST_FILE} SYMBOLIC)

    list(APPEND SYMLINK_PYTEST_FILES ${SYMLINK_PYTEST_FILE})
endforeach()

add_custom_target(
  tests_py
  COMMAND pytest ${SYMLINK_PYTEST_FILES}
  DEPENDS ${WDR_PYTHON_PACKAGE_NAME}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/python
  USES_TERMINAL
)

add_custom_target(tests DEPENDS tests_cpp tests_py)

add_subdirectory(wanderary)
add_subdirectory(python)
