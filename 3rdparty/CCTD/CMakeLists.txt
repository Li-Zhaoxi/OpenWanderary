file (GLOB_RECURSE HEADERS "*.h")
file (GLOB_RECURSE SOURCES "[^_]*.cpp")


include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(LIBRARY cctd)
add_library(${LIBRARY} SHARED ${SOURCES})
# set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs)
