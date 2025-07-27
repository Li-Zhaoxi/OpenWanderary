file (GLOB_RECURSE SOURCES "*.cpp")
set(LIBRARY wdr_data_loader)
add_library(${LIBRARY} SHARED ${SOURCES})
target_link_libraries(${LIBRARY} tbb wdr_structs ${OpenCV_LIBS})
