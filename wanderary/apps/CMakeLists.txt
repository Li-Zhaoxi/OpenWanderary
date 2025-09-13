file (GLOB_RECURSE SOURCES "*.cpp")



set(LIBRARY wdr_apps)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} -lglog -lopencv_core wdr_utils wdr_dnn wdr_process -ldnn -lcnn_intf -lhbrt_bayes_aarch64)
