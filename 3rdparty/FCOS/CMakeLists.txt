file (GLOB_RECURSE HEADERS "*.h")
file (GLOB_RECURSE SOURCES "[^_]*.cpp" "[^_]*.cc")


include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(LIBRARY fcos)
add_library(${LIBRARY} SHARED ${SOURCES})
# set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} 
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
-lglog -lgflags -lpthread -lrt -dl
/usr/lib/libdnn.so
/usr/hobot/lib/libcnn_intf.so
# /usr/lib/hbbpu/libhbrt_bernoulli_aarch64.so
/usr/lib/libhbrt_bayes_aarch64.so
)
