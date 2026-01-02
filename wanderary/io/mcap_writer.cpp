#include "wanderary/io/mcap_writer.h"

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "wanderary/io/msg_convertor/foxglove.h"
#include "wanderary/io/msg_convertor/open_waymo_dataset.h"
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
                            const std::string& image_path, uint32_t sequence) {
  foxglove::CompressedImage msg;
  const int64_t cur_ts = wdr::CurrentMilliseconds();
  wdr::msg::ConvertImageToMsg(image_path, cur_ts, random_image_frame_id, &msg);
  this->write<foxglove::CompressedImage>(topic_name, msg, cur_ts * 1e6,
                                         cur_ts * 1e6, sequence);
}

bool MCAPWriter::WriteImage(const std::string& topic_name,
                            const ImageFrame& frame, uint32_t sequence) {
  foxglove::CompressedImage msg;
  const int64_t cur_ts = frame.start_timestamp;
  if (!frame.meta.image_file.has_value()) return false;
  const auto& image_file = frame.meta.image_file.value();
  const std::string frame_id = SensorNameID2str(frame.sensor_name_id);
  if (image_file.rawpath.empty()) {
    if (image_file.data == nullptr) return false;
    wdr::msg::ConvertImageToMsg(*image_file.data, cur_ts, frame_id, &msg);
  } else {
    wdr::msg::ConvertImageToMsg(image_file.rawpath, cur_ts, frame_id, &msg);
  }

  this->write<foxglove::CompressedImage>(topic_name, msg, cur_ts * 1e6,
                                         cur_ts * 1e6, sequence);

  return true;
}

void MCAPWriter::WriteWaymoFrame(const std::string& topic_name,
                                 const std::vector<uint8_t>& bytes,
                                 uint32_t sequence, MultiModalFrame* mmframe) {
  waymo::open_dataset::Frame frame;
  CHECK(frame.ParseFromArray(bytes.data(), bytes.size()))
      << "failed to parse tfrecord data block into a Frame";

  if (!topic_name.empty()) {
    // 如果Topic不为空, 写入原始数据
    const int64_t cur_ts = frame.timestamp_micros();
    this->write<waymo::open_dataset::Frame>(topic_name, frame, cur_ts * 1e3,
                                            cur_ts * 1e3, sequence);
  }

  if (mmframe) {
    wdr::msg::ConvertWaymoFrameMsgToMMFrame(frame, mmframe);
  }
}

}  // namespace wdr::io
