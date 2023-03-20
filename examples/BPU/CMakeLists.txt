

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")


link_directories(
  /usr/lib/hbbpu/
)

set(BIN infer_unet)
add_executable(${BIN} infer_unet.cpp)
set_property(TARGET ${BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
target_link_libraries(${BIN} 
-lopencv_imgcodecs
openwdr_core openwdr_bpu openwdr_dnn)