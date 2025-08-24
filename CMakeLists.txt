cmake_minimum_required(VERSION 3.5)

project(wanderary LANGUAGES CXX)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(WDR_PYTHON_PACKAGE_NAME "${PROJECT_NAME}")

set(WDR_PYTHON_PACKAGE_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
set(WDR_PROJECT_VERSION ${WDR_PYTHON_PACKAGE_VERSION})

message(STATUS "Python package name: ${WDR_PYTHON_PACKAGE_NAME}, version: ${WDR_PYTHON_PACKAGE_VERSION}")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)


find_package(gflags REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Boost 1.71.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(pybind11 REQUIRED)
find_package(pybind11_json REQUIRED)
find_package(TBB REQUIRED)
find_package(yaml-cpp REQUIRED)
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "PyBind11 Dir: ${pybind11_DIR}")



message(STATUS "OPENCV VERSION: ${OpenCV_VERSION}")

include("${CMAKE_SOURCE_DIR}/cmake/pack_3rdparty.cmake")

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
)

message(STATUS "OPENCV Lib path: ${OpenCV_LIBRARY_DIRS}")
link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

link_directories(
  /usr/hobot/lib/
)

message( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
message( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
message( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")

add_subdirectory(3rdparty)

add_subdirectory(wanderary)
add_subdirectory(examples)
add_subdirectory(tests)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/wdr-config-version.cmake"
    VERSION ${WDR_PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
  TARGETS wdr_utils wdr_structs wdr_media wdr_dnn wdr_testing wdr_process wdr_data_loader
  EXPORT wdr-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/wanderary/
  DESTINATION include/wanderary
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN "python" EXCLUDE
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/wdr-config.cmake.in"
    "${PROJECT_BINARY_DIR}/wdr-config.cmake"
    INSTALL_DESTINATION lib/cmake/wanderary
)

install(EXPORT wdr-targets
  NAMESPACE wdr::
  DESTINATION lib/cmake/wanderary)

install(
  FILES "${PROJECT_BINARY_DIR}/wdr-config-version.cmake"
        "${PROJECT_BINARY_DIR}/wdr-config.cmake"
  DESTINATION lib/cmake/wanderary
)

# 生成卸载脚本
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake_uninstall.cmake.in
  ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
  IMMEDIATE @ONLY
)

# 添加 uninstall 目标
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
)
