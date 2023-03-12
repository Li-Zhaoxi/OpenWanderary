

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")


link_directories(
  /usr/lib/hbbpu/
)

set(LIBRARY openwdr_bpu) 
add_library(${LIBRARY} SHARED ${SOURCES})
set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

target_link_libraries(${LIBRARY} -lpthread -lrt -dl 
-ldnn -lcnn_intf -lhbrt_bernoulli_aarch64 
-lglog -lgflags 
-lopencv_core -lopencv_highgui -lopencv_imgproc
openwdr_core)

