
include_directories(
  ${CMAKE_SOURCE_DIR}/wanderary/
  ${CMAKE_SOURCE_DIR}/3rdparty/
)
 
add_subdirectory(Core)
add_subdirectory(DNN)
add_subdirectory(BPU)
add_subdirectory(Algorithms)