cmake_minimum_required(VERSION 3.5)
project(BPUAuxiliaryTools)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -Wno-write-strings -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.0)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()

find_package(gflags REQUIRED)

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
MESSAGE(${OpenCV_VERSION})

include_directories(
${PROJECT_SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
########### Build Algorithm Libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# build common
add_library(common SHARED 
src/common.cpp)
target_link_libraries(common -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
-lglog -lgflags -lpthread -lrt -dl -lglog -lgflags
/usr/lib/hbbpu/libdnn.so
/usr/lib/hbbpu/libcnn_intf.so
/usr/lib/hbbpu/libhbrt_bernoulli_aarch64.so)

# build YoloV5
add_library(yolov5 SHARED 
src/YOLO/yolov5.cpp)
target_link_libraries(yolov5 common)

#### Build Unit Examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/examples)

# build infer_yolov5
add_executable(infer_yolov5 examples/infer_yolov5.cpp)
target_link_libraries(infer_yolov5 yolov5)