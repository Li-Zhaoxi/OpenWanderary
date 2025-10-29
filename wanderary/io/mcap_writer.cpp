#include "wanderary/io/mcap_writer.h"

#include <queue>
#include <string>
#include <unordered_set>

#include "wanderary/io/msg_convertor/foxglove.h"
#include "wanderary/utils/time_manager.h"

namespace wdr::io {

MCAPWriter::MCAPWriter(const std::string& filepath, uint64_t chunk_size,
                       mcap::Compression compression, bool enable_crcs) {
  mcap::McapWriterOptions options("");
  options.chunkSize = chunk_size;
  options.compression = compression;
  options.enableDataCRC = enable_crcs;
  auto status = this->writer_.open(filepath, options);

  if (!status.ok())
    LOG(FATAL) << "Failed to open MCAP file: " << status.message;
  closed_ = false;
}

google::protobuf::FileDescriptorSet MCAPWriter::BuildFileDescriptorSet(
    const google::protobuf::Descriptor* toplevelDescriptor) {
  google::protobuf::FileDescriptorSet fdSet;
  std::queue<const google::protobuf::FileDescriptor*> toAdd;
  toAdd.push(toplevelDescriptor->file());
  std::unordered_set<std::string> seenDependencies;
  while (!toAdd.empty()) {
    const google::protobuf::FileDescriptor* next = toAdd.front();
    toAdd.pop();
    next->CopyTo(fdSet.add_file());
    for (int i = 0; i < next->dependency_count(); ++i) {
      const auto& dep = next->dependency(i);
      if (seenDependencies.find(dep->name()) == seenDependencies.end()) {
        seenDependencies.insert(dep->name());
        toAdd.push(dep);
      }
    }
  }
  return fdSet;
}

void MCAPWriter::close() {
  this->writer_.close();
  closed_ = true;
}

MCAPWriter::~MCAPWriter() {
  if (!closed_) close();
}

void MCAPWriter::WriteImage(const std::string& topic_name,
                            const std::string& image_path) {
  foxglove::CompressedImage msg;
  const int64_t cur_ts = wdr::CurrentMilliseconds();
  wdr::msg::ConvertMsgImage(image_path, cur_ts, random_image_frame_id, &msg);
  this->write<foxglove::CompressedImage>(topic_name, msg, cur_ts * 1e6,
                                         cur_ts * 1e6, 0);
}

}  // namespace wdr::io
