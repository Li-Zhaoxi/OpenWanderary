#include "wanderary/io/msg_convertor/protobuf.h"

namespace wdr::msg {

void ConvertTimestampToMsg(int64_t timestamp,
                           google::protobuf::Timestamp* msg) {
  msg->set_seconds(timestamp / 1000);
  msg->set_nanos(timestamp % 1000 * 1000000);
}

int64_t ConvertTimestampFromMsg(const google::protobuf::Timestamp& msg) {
  return msg.seconds() * 1000 + msg.nanos() / 1000000;
}

}  // namespace wdr::msg
