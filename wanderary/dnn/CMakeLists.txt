file (GLOB_RECURSE SOURCES "*.cpp")



set(LIBRARY wdr_dnn)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} -lglog -lopencv_core wdr_utils -ldnn bpu::cnn_intf -lhbrt_bayes_aarch64)
