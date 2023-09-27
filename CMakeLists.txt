cmake_minimum_required(VERSION 3.5)
project(wanderary)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set (CMAKE_CXX_STANDARD 17)
# # Check C++11 or C++0x support
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#    add_definitions(-DCOMPILEDWITHC11)
#    message(STATUS "Using flag -std=c++11.")
# elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#    add_definitions(-DCOMPILEDWITHC0X)
#    message(STATUS "Using flag -std=c++0x.")
# else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

find_package(gflags REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Boost 1.71.0 REQUIRED)
find_package(Eigen3 REQUIRED)

MESSAGE("OPENCV VERSION: ${OpenCV_VERSION}")

 
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
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