file(GLOB SOURCES "[^_]*.cpp")

pybind11_add_module(${WDR_PYTHON_PACKAGE_NAME} ${SOURCES})
target_compile_definitions(${WDR_PYTHON_PACKAGE_NAME} PRIVATE)
target_link_libraries(${WDR_PYTHON_PACKAGE_NAME} PRIVATE wdr_process wdr_utils wdr_media  pybind11_json)
