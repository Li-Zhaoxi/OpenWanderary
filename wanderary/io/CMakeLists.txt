file (GLOB_RECURSE SOURCES "*.cpp")

set(LIBRARY wdr_io)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} mcap::mcap -lglog ${Protobuf_LIBRARIES} wdr_utils wdr_structs wdr_proto_foxglove wdr_proto_waymo wdr_proto_tf)
