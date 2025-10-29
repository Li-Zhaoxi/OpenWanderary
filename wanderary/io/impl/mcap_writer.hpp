#pragma once
#include <string>
#include <vector>

#include <glog/logging.h>
#include <wanderary/io/mcap_writer.h>
#include <wanderary/utils/time_manager.h>

namespace wdr::io {

template <class PBMessageType>
const RegisterTopic &MCAPWriter::register_topic(const std::string &topic_name) {
  auto iter = this->registered_topics_.find(topic_name);

  if (iter == this->registered_topics_.end()) {
    RegisterTopic reg_topic;

    // 注册Schema
    mcap::Schema schema(
        PBMessageType::descriptor()->full_name(), "protobuf",
        this->BuildFileDescriptorSet(PBMessageType::descriptor())
            .SerializeAsString());
    this->writer_.addSchema(schema);

    // 注册Channel
    mcap::Channel channel(topic_name, "protobuf", schema.id);
    this->writer_.addChannel(channel);

    reg_topic.schema_id = schema.id;
    reg_topic.schema_name = schema.name;
    reg_topic.channel_id = channel.id;
    registered_topics_[topic_name] = reg_topic;

    iter = this->registered_topics_.find(topic_name);
  }

  return iter->second;
}

template <class PBMessageType>
void MCAPWriter::write(const std::string &topic_name, const PBMessageType &msg,
                       int64_t log_time, int64_t publish_time,
                       uint32_t sequence) {
  const std::string schema_name = PBMessageType::descriptor()->full_name();
  const auto &reg_topic = this->register_topic<PBMessageType>(topic_name);
  CHECK_EQ(reg_topic.schema_name, schema_name);

  std::vector<std::byte> payload(msg.ByteSizeLong());
  const size_t nbytes = payload.size();
  if (nbytes > 0) msg.SerializeToArray(payload.data(), nbytes);

  mcap::Message mcap_msg;
  mcap_msg.channelId = reg_topic.channel_id;
  mcap_msg.sequence = sequence;

  mcap_msg.logTime = log_time < 0 ? wdr::CurrentMilliseconds() : log_time;
  mcap_msg.publishTime =
      publish_time < 0 ? wdr::CurrentMilliseconds() : publish_time;

  mcap_msg.data = payload.data();
  mcap_msg.dataSize = nbytes;

  auto status = this->writer_.write(mcap_msg);
  if (!status.ok()) LOG(FATAL) << "Failed to write message: " << status.message;
}

}  // namespace wdr::io
