

add_library(hobot::multimedia SHARED IMPORTED GLOBAL)
set_target_properties(hobot::multimedia PROPERTIES
IMPORTED_LOCATION "/usr/hobot/lib/libmultimedia.so"
INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
)

add_library(bpu::cnn_intf SHARED IMPORTED GLOBAL)
set_target_properties(bpu::cnn_intf PROPERTIES
  IMPORTED_LOCATION "/usr/hobot/lib/libcnn_intf.so"
)
