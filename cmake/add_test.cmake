macro(WDR_ADD_TEST _exename _folder)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs ARGUMENTS LINK_WITH APPEND_FOLDERS)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments to WDR_ADD_TEST: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  aux_source_directory(${_folder} DIR_SRC_TESTS)

  if (ARGS_APPEND_FOLDERS)
    foreach (FOLDER ${ARGS_APPEND_FOLDERS})
      aux_source_directory(${FOLDER} DIR_SRC)
      list(APPEND DIR_SRC_TESTS ${DIR_SRC})
    endforeach()
  endif()

  add_executable(${_exename} ${DIR_SRC_TESTS})
  target_link_libraries(${_exename} ${ARGS_LINK_WITH} GTest::gtest_main)

  gtest_discover_tests(${_exename} PROPERITIES TEST_DISCOVERY_TIMEOUT 120)

  add_dependencies(tests_cpp ${_exename})
endmacro()
