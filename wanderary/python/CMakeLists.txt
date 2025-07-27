file(GLOB SOURCES "[^_]*.cpp")

pybind11_add_module(${WDR_PYTHON_PACKAGE_NAME} ${SOURCES})
target_compile_definitions(${WDR_PYTHON_PACKAGE_NAME} PRIVATE)
target_link_libraries(${WDR_PYTHON_PACKAGE_NAME} PRIVATE wdr_process wdr_utils wdr_media wdr_dnn wdr_structs wdr_data_loader wdr_testing pybind11_json)
