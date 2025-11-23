#pragma once
#include <google/protobuf/timestamp.pb.h>

namespace wdr::msg {

// 单位ms
void ConvertTimestampToMsg(int64_t timestamp, google::protobuf::Timestamp* msg);

int64_t ConvertTimestampFromMsg(const google::protobuf::Timestamp& msg);

}  // namespace wdr::msg
