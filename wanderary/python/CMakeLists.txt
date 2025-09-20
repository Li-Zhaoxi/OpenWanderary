file(GLOB SOURCES "[^_]*.cpp")

pybind11_add_module(${WDR_PYTHON_PACKAGE_NAME} ${SOURCES})
target_compile_definitions(${WDR_PYTHON_PACKAGE_NAME} PRIVATE)
target_link_libraries(${WDR_PYTHON_PACKAGE_NAME} PRIVATE wdr_process wdr_utils wdr_media wdr_dnn wdr_structs wdr_data_loader wdr_testing pybind11_json wdr_visualization wdr_apps)

# 将生成的模块路径写入文件
set(PYMODULE_PATH ${CMAKE_BINARY_DIR}/py_module_path.txt)
file(GENERATE OUTPUT ${PYMODULE_PATH} CONTENT "$<TARGET_FILE:${WDR_PYTHON_PACKAGE_NAME}>" )

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/py-config.yaml
  ${CMAKE_BINARY_DIR}/py-config.yaml
  @ONLY
)

add_custom_target(
  wheel ALL
  COMMAND python ../setup.py clean --all && python ../setup.py bdist_wheel
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Build python wheel"
  DEPENDS ${WDR_PYTHON_PACKAGE_NAME}
)
