

#include <string>

#include <glog/logging.h>

#include "wanderary/media/media_codec.h"
namespace wdr::media {

MediaCodecID CodecDescriptor::id() const {
  CHECK(this->has_value());
  return int2MediaCodecID(this->desc_->id);
}

MediaCodecMode CodecDescriptor::mode() const {
  CHECK(this->has_value());
  return int2MediaCodecMode(this->desc_->mode);
}

std::string CodecDescriptor::name() const {
  CHECK(this->has_value());
  return std::string(this->desc_->name);
}

std::string CodecDescriptor::long_name() const {
  CHECK(this->has_value());
  return std::string(this->desc_->long_name);
}

std::string CodecDescriptor::mime_types() const {
  CHECK(this->has_value());
  return std::string(this->desc_->mime_types);
}

}  // namespace wdr::media
