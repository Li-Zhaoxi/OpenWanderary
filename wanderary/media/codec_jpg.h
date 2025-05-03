#pragma once

#include <wanderary/media/codec_base.h>

namespace wdr::media {

class MediaCodecJpg : public MediaCodecBase {
 public:
  MediaCodecJpg(MediaCodecID codec_id, bool encode, int width, int height,
                CodecPixelFormat pixfmt);
  virtual ~MediaCodecJpg() = default;
};

}  // namespace wdr::media
