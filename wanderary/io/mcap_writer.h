#pragma once
#include <map>
#include <set>
#include <string>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>

#include <mcap/writer.hpp>

namespace wdr::io {

struct RegisterTopic {
  mcap::SchemaId schema_id;
  std::string schema_name;
  mcap::ChannelId channel_id;
};

class MCAPWriter {
 public:
  MCAPWriter(const std::string& filepath, uint64_t chunk_size = 1024 * 1024,
             mcap::Compression compression = mcap::Compression::Zstd,
             bool enable_crcs = true);
  ~MCAPWriter();
  template <class PBMessageType>
  const RegisterTopic& register_topic(const std::string& topic_name);

  template <class PBMessageType>
  void write(const std::string& topic_name, const PBMessageType& msg,
             int64_t log_time = -1, int64_t publish_time = -1,
             uint32_t sequence = 0);

  void close();

  // 写入图像
  void WriteImage(const std::string& topic_name, const std::string& image_path);

 private:
  static google::protobuf::FileDescriptorSet BuildFileDescriptorSet(
      const google::protobuf::Descriptor* toplevelDescriptor);

  mcap::McapWriter writer_;
  bool closed_;
  std::map<std::string, RegisterTopic> registered_topics_;

  static constexpr char random_image_frame_id[] = "random";
};

}  // namespace wdr::io

#include <wanderary/io/impl/mcap_writer.hpp>
