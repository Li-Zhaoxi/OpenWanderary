

add_library(LZ4::lz4 SHARED IMPORTED GLOBAL)
set_target_properties(LZ4::lz4 PROPERTIES
IMPORTED_LOCATION "/lib/aarch64-linux-gnu/liblz4.so"
INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
)

add_library(zstd::libzstd SHARED IMPORTED GLOBAL)
set_target_properties(zstd::libzstd PROPERTIES
  IMPORTED_LOCATION "/lib/aarch64-linux-gnu/libzstd.so"
  INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
)
