
set(WAYMO_SRC_DIR ${PROJECT_SOURCE_DIR}/3rdparty/waymo-open-dataset/src/waymo_open_dataset)
set(WAYMO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/waymo-open-dataset/src)

list(APPEND WAYMO_SOURCES
  ${WAYMO_SRC_DIR}/common/status.cc
  ${WAYMO_SRC_DIR}/common/status.h
  ${WAYMO_SRC_DIR}/math/polygon2d.cc
  ${WAYMO_SRC_DIR}/math/polygon2d.h
  ${WAYMO_SRC_DIR}/math/box2d.cc
  ${WAYMO_SRC_DIR}/math/box2d.h
  ${WAYMO_SRC_DIR}/math/vec2d.h
  ${WAYMO_SRC_DIR}/math/math_util.h
  ${WAYMO_SRC_DIR}/metrics/config_util.cc
  ${WAYMO_SRC_DIR}/metrics/config_util.h
  ${WAYMO_SRC_DIR}/metrics/matcher.cc
  ${WAYMO_SRC_DIR}/metrics/matcher.h
  ${WAYMO_SRC_DIR}/metrics/iou.cc
  ${WAYMO_SRC_DIR}/metrics/iou.h
  ${WAYMO_SRC_DIR}/metrics/mot.cc
  ${WAYMO_SRC_DIR}/metrics/mot.h
  ${WAYMO_SRC_DIR}/metrics/hungarian.cc
  ${WAYMO_SRC_DIR}/metrics/hungarian.h
  ${WAYMO_SRC_DIR}/metrics/breakdown_generator.cc
  ${WAYMO_SRC_DIR}/metrics/breakdown_generator.h
  ${WAYMO_SRC_DIR}/metrics/metrics_utils.cc
  ${WAYMO_SRC_DIR}/metrics/metrics_utils.h
  ${WAYMO_SRC_DIR}/metrics/motion_metrics_utils.cc
  ${WAYMO_SRC_DIR}/metrics/motion_metrics_utils.h
  ${WAYMO_SRC_DIR}/metrics/motion_metrics.cc
  ${WAYMO_SRC_DIR}/metrics/motion_metrics.h
  ${WAYMO_SRC_DIR}/metrics/tracking_metrics.cc
  ${WAYMO_SRC_DIR}/metrics/tracking_metrics.h
  ${WAYMO_SRC_DIR}/metrics/segmentation_metrics.cc
  ${WAYMO_SRC_DIR}/metrics/segmentation_metrics.h
  ${WAYMO_SRC_DIR}/metrics/detection_metrics.cc
  ${WAYMO_SRC_DIR}/metrics/detection_metrics.h
)

set(LIBRARY wdr_waymo_tools)
add_library(${LIBRARY} SHARED ${WAYMO_SOURCES})
target_link_libraries(${LIBRARY} wdr_proto_waymo -lglog absl::strings absl::str_format_internal)
target_include_directories(${LIBRARY} PUBLIC ${WAYMO_INCLUDE_DIR})

set(BIN compute_detection_let_metrics_main)
add_executable(${BIN} ${WAYMO_SRC_DIR}/metrics/tools/compute_detection_let_metrics_main.cc)
target_link_libraries(${BIN} wdr_proto_waymo wdr_waymo_tools -lglog absl::flat_hash_map)


set(BIN compute_detection_metrics_main)
add_executable(${BIN} ${WAYMO_SRC_DIR}/metrics/tools/compute_detection_metrics_main.cc)
target_link_libraries(${BIN} wdr_proto_waymo wdr_waymo_tools -lglog absl::flat_hash_map)

set(BIN compute_segmentation_metrics_main)
add_executable(${BIN} ${WAYMO_SRC_DIR}/metrics/tools/compute_segmentation_metrics_main.cc)
target_link_libraries(${BIN} wdr_proto_waymo wdr_waymo_tools -lglog absl::flat_hash_map ZLIB::ZLIB)

set(BIN compute_tracking_metrics_main)
add_executable(${BIN} ${WAYMO_SRC_DIR}/metrics/tools/compute_tracking_metrics_main.cc)
target_link_libraries(${BIN} wdr_proto_waymo wdr_waymo_tools -lglog absl::flat_hash_map)
