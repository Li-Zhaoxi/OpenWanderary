# =================================================================================================
#  Copyright (C) 2023-2024 MCAP_BUILDER Contributors
# =================================================================================================

cmake_minimum_required(VERSION 3.22.1)

project(mcap_builder LANGUAGES CXX)
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 4)
set(PROJECT_VERSION_PATCH 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(MCAP_PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

include("${CMAKE_SOURCE_DIR}/cmake/pack_3rdparty.cmake")

add_subdirectory(src)


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mcap-config-version.cmake"
    VERSION ${MCAP_PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)


install(DIRECTORY ${PROJECT_SOURCE_DIR}/../mcap/cpp/mcap/include/mcap/
  DESTINATION include/mcap
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(TARGETS mcap EXPORT mcap-targets)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/mcap-config.cmake.in"
    "${PROJECT_BINARY_DIR}/mcap-config.cmake"
    INSTALL_DESTINATION lib/cmake/mcap
)

install(EXPORT mcap-targets
  NAMESPACE mcap::
  DESTINATION lib/cmake/mcap)

install(
  FILES "${PROJECT_BINARY_DIR}/mcap-config-version.cmake"
        "${PROJECT_BINARY_DIR}/mcap-config.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/pack_3rdparty.cmake"
  DESTINATION lib/cmake/mcap
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
