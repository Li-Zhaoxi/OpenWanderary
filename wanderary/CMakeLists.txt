
include_directories(
  ${CMAKE_SOURCE_DIR}/wanderary/
  ${CMAKE_SOURCE_DIR}/3rdparty/
)


add_subdirectory(utils)
add_subdirectory(structs)
add_subdirectory(data_loader)
add_subdirectory(process)
add_subdirectory(media)
add_subdirectory(dnn)
add_subdirectory(Core)
add_subdirectory(BPU)
add_subdirectory(Algorithms)
add_subdirectory(testing)
add_subdirectory(python)
add_subdirectory(visualization)
add_subdirectory(apps)
