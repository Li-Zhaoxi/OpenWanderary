file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})


set(LIBRARY cnpy)
add_library(${LIBRARY} SHARED ${SOURCES})
# set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} ${ZLIB_LIBRARIES})
