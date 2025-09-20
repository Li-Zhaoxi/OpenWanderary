file (GLOB_RECURSE SOURCES "*.cpp")

set(LIBRARY wdr_structs)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} wdr_utils -lglog)
