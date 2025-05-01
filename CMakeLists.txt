cmake_minimum_required(VERSION 3.5)
project(wanderary)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)


find_package(gflags REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Boost 1.71.0 REQUIRED)
find_package(Eigen3 REQUIRED)

MESSAGE("OPENCV VERSION: ${OpenCV_VERSION}")


include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
)

message("OPENCV Lib path: ${OpenCV_LIBRARY_DIRS}")
link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")

MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")

MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")

add_subdirectory(3rdparty)

add_subdirectory(wanderary)
add_subdirectory(examples)
