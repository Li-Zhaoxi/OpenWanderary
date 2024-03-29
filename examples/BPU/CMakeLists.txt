

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")


link_directories(
  /usr/lib/hbbpu/
)

set(BIN infer-unet)
add_executable(${BIN} infer-unet.cpp)
target_link_libraries(${BIN} 
${OpenCV_LIBS}
openwdr_core openwdr_bpu openwdr_dnn cnpy)

set(BIN infer-yolov5)
add_executable(${BIN} infer-yolov5.cpp)
target_link_libraries(${BIN} 
${OpenCV_LIBS}
openwdr_core openwdr_bpu openwdr_dnn cnpy)


set(BIN print-infos)
add_executable(${BIN} print-infos.cpp)
target_link_libraries(${BIN} 
${OpenCV_LIBS} openwdr_core openwdr_bpu)


set(BIN infer_unet_standalong)
add_executable(${BIN} infer_unet_standalong.cpp)
target_link_libraries(${BIN} 
-lpthread -lrt -dl 
/usr/lib/hbbpu/libdnn.so
/usr/lib/hbbpu/libcnn_intf.so
/usr/lib/hbbpu/libhbrt_bernoulli_aarch64.so
-lglog -lgflags 
${OpenCV_LIBS} cnpy)

# -ldnn -lcnn_intf -lhbrt_bernoulli_aarch64 

include_directories(
  ${CMAKE_SOURCE_DIR}/3rdparty/
  ${CMAKE_SOURCE_DIR}/3rdparty/FCOS
  )
set(BIN infer_fcos)
add_executable(${BIN} infer_fcos.cpp)
target_link_libraries(${BIN} ${OpenCV_LIBS}
fcos)


set(BIN infer_demo)
add_executable(${BIN} infer_demo.cpp)
target_link_libraries(${BIN} 
${OpenCV_LIBS}
-lglog -lgflags -lpthread -lrt -dl
/usr/lib/hbbpu/libdnn.so
/usr/lib/hbbpu/libcnn_intf.so
/usr/lib/hbbpu/libhbrt_bernoulli_aarch64.so)