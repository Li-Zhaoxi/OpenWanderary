#include "wanderary/structs/frame.h"

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

}  // namespace wdr
