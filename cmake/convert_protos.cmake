

function(wdr_proto_library)
  set(oneValueArgs NAME INPUT_DIR OUTPUT_DIR)

  set(multiValueArgs FOLDERS)
  cmake_parse_arguments(PROTO_LIB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT PROTO_LIB_FOLDERS)
    file(GLOB_RECURSE ALL_PROTOS "${PROTO_LIB_INPUT_DIR}/*.proto")
  else()
    foreach(f ${PROTO_LIB_FOLDERS})
      file(GLOB_RECURSE FOLDER_PROTOS "${PROTO_LIB_INPUT_DIR}/${f}/*.proto")
      list(APPEND ALL_PROTOS ${FOLDER_PROTOS})
    endforeach()
  endif()

  foreach(f ${ALL_PROTOS})
    file(RELATIVE_PATH f ${PROTO_LIB_INPUT_DIR} ${f})
    string(REGEX REPLACE "\\.proto$" "" PROTO_PREFIX ${f})

    list(APPEND PROTO_HEADERS "${PROTO_LIB_OUTPUT_DIR}/${PROTO_PREFIX}.pb.h")
    list(APPEND PROTO_SOURCES "${PROTO_LIB_OUTPUT_DIR}/${PROTO_PREFIX}.pb.cc")
  endforeach(f)

  add_custom_command(
    OUTPUT ${PROTO_SOURCES} ${PROTO_HEADERS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE} --proto_path=${PROTO_LIB_INPUT_DIR} --cpp_out=${PROTO_LIB_OUTPUT_DIR} ${ALL_PROTOS}
  )

  add_library(${PROTO_LIB_NAME} SHARED ${PROTO_SOURCES})
  target_link_libraries(${PROTO_LIB_NAME} INTERFACE ${Protobuf_LIBRARIES})
endfunction(wdr_proto_library)
