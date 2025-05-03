#pragma once

#include <wanderary/media/codec_base.h>

namespace wdr::media {

class MediaCodecJpg : public MediaCodecBase {
 public:
  MediaCodecJpg(MediaCodecID codec_id, bool encode, int width, int height,
                CodecPixelFormat pixfmt);
  virtual ~MediaCodecJpg() = default;

 protected:
  void check_valid(const cv::Mat &frame) const override;
};

}  // namespace wdr::media
