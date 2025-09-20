file (GLOB_RECURSE SOURCES "*.cpp")



set(LIBRARY wdr_apps)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} PUBLIC -lglog -lopencv_core wdr_utils wdr_structs wdr_dnn wdr_process -ldnn bpu::cnn_intf -lhbrt_bayes_aarch64)
