#include "wanderary/structs/frame.h"

#include <map>
#include <utility>

#include "wanderary/utils/common_utils.h"
namespace wdr {

Frame::Metas Frame::Metas::clone() const {
  Frame::Metas metas;
  if (image_file.has_value()) metas.image_file = image_file->clone();
  return metas;
}

Frame Frame::clone() const {
  Frame frame;
  frame.start_timestamp = start_timestamp;
  frame.instances = instances;
  frame.meta = meta.clone();
  return frame;
}

void MultiModalFrame::add(ImageFrame &&frame) {
  CHECK(SensorUtils::is_camera(frame.sensor_name_id));
  CHECK(!wdr::contains(camera_frames_, frame.sensor_name_id));
  camera_frames_[frame.sensor_name_id] = std::move(frame);
}

std::optional<std::reference_wrapper<ImageFrame>>
MultiModalFrame::mutable_camera_frame(SensorNameID name_id) {
  if (wdr::contains(camera_frames_, name_id)) {
    return std::ref(camera_frames_[name_id]);
  }
  return std::nullopt;
}

std::map<SensorNameID, std::reference_wrapper<const ImageFrame>>
MultiModalFrame::camera_frames() const {
  std::map<SensorNameID, std::reference_wrapper<const ImageFrame>> frames;
  for (const auto &frame : camera_frames_)
    frames.insert(std::make_pair(frame.first, std::cref(frame.second)));
  return frames;
}

}  // namespace wdr
