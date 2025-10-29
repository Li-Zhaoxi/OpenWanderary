
# 转换foxglove的proto文件
set(FOXGLOVE_INPUT_DIR "${CMAKE_SOURCE_DIR}/3rdparty/mcap/cpp/examples/protobuf/proto")
wdr_proto_library(
  NAME wdr_proto_foxglove
  INPUT_DIR ${FOXGLOVE_INPUT_DIR}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/
)

# 转换foxglove的proto文件
set(WAYMO_INPUT_DIR "${CMAKE_SOURCE_DIR}/3rdparty/waymo-open-dataset/src")
wdr_proto_library(
  NAME wdr_proto_waymo
  INPUT_DIR ${WAYMO_INPUT_DIR}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/
)
