#include "wanderary/io/mcap_reader.h"

#include <string>

#include <glog/logging.h>

#include "wanderary/io/msg_convertor/foxglove.h"

namespace wdr::io {

MCAPReader::MCAPReader(const std::string &filename) {
  mcap::Status status;
  status = this->reader_.open(filename);
  CHECK(status.ok()) << status.message;

  status = this->reader_.readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
  CHECK(status.ok()) << status.message;

  const auto stats = reader_.statistics();
  if (stats.has_value()) {
    LOG(INFO) << "Channel count: " << stats->channelMessageCounts.size();
    for (const auto &[chlid, count] : stats->channelMessageCounts) {
      LOG(INFO) << "Channel " << chlid << " has " << count
                << " messages, channel name: " << reader_.channel(chlid)->topic;
    }
  }

  // 统计每个通道的logtime，以便后续处理. 更好的处理是获取offset，后面完善这里
  auto messageView = reader_.readMessages();
  for (auto it = messageView.begin(); it != messageView.end(); it++) {
    this->channel_times_[it->channel->topic].push_back(it->message.logTime);
  }

  is_open_ = true;
}

void MCAPReader::close() {
  if (is_open_) {
    reader_.close();
    is_open_ = false;
  }
}

MCAPReader::~MCAPReader() { this->close(); }

int MCAPReader::size(const std::string &topic) const {
  return this->channel_times_.at(topic).size();
}

int64_t MCAPReader::ReadImage(const std::string &topic, int index,
                              cv::Mat *data, bool decode) {
  const auto iter = this->channel_times_.find(topic);
  CHECK(iter != this->channel_times_.end())
      << "Topic " << topic << " not found";
  const int64_t dst_time = iter->second[index];

  mcap::ReadMessageOptions opt;
  opt.startTime = dst_time;
  opt.endTime = dst_time + 1;
  opt.topicFilter = [&topic](std::string_view tp) -> bool {
    return tp == topic;
  };

  int64_t ts = -1;
  auto messageView = reader_.readMessages([](const mcap::Status &) {}, opt);
  for (auto it = messageView.begin(); it != messageView.end(); it++) {
    CHECK_EQ(it->schema->name, "foxglove.CompressedImage");

    foxglove::CompressedImage msg;
    CHECK(msg.ParseFromArray(it->message.data,
                             static_cast<int>(it->message.dataSize)));

    std::string format;
    ts = wdr::msg::ConvertImageFromMsg(msg, decode, data, &format);
    break;
  }

  return ts;
}

}  // namespace wdr::io
