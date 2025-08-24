

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")


link_directories(
  /usr/lib/
  /usr/hobot/lib/
)

set(BIN stat-dnn-time-consumption)
add_executable(${BIN} stat-dnn-time-consumption.cpp)
target_link_libraries(${BIN} -lgflags
${OpenCV_LIBS} wdr_dnn wdr_process wdr_data_loader wdr_visualization)
