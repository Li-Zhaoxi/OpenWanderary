file (GLOB_RECURSE SOURCES "*.cpp")

set(LIBRARY wdr_utils)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} -lglog )
