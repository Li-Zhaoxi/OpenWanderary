file (GLOB_RECURSE SOURCES "*.cpp")

set(LIBRARY wdr_visualization)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} -lopencv_core wdr_structs)
