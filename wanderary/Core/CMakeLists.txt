

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp") 



set(LIBRARY openwdr_core)
add_library(${LIBRARY} SHARED ${SOURCES})
# set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} -lpthread -lrt -dl -lglog -lgflags -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs)

