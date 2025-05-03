#pragma once

#include <wanderary/media/media_codec.h>

namespace wdr::media {

class MediaCodecBase {
 public:
  MediaCodecBase();
  virtual ~MediaCodecBase() = default;

  void Init();
  void Close();

 protected:
  CodecContext ctx_;
  CodecStartupParams params_;
};

}  // namespace wdr::media
