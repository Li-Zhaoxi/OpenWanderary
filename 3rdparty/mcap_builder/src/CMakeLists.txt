add_library(mcap SHARED lib.cpp)

target_include_directories(mcap PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../mcap/cpp/mcap/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mcap PUBLIC LZ4::lz4 zstd::libzstd)
