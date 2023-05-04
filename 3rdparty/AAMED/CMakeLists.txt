file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")





set(LIBRARY aamed)
add_library(${LIBRARY} SHARED ${SOURCES})
# set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_flann)
