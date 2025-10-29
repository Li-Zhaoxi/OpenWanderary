#include <google/protobuf/timestamp.pb.h>

namespace wdr::msg {

// 单位ms
void ConvertTimestampMsg(int64_t timestamp, google::protobuf::Timestamp* msg);

}  // namespace wdr::msg
