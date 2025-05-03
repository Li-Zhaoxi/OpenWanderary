#pragma once

#include <wanderary/media/media_codec.h>

#include <opencv2/opencv.hpp>

namespace wdr::media {

class MediaCodecBase {
 public:
  MediaCodecBase(MediaCodecID codec_id, bool encode)
      : params_(CodecStartupParams(codec_id, encode)) {}
  virtual ~MediaCodecBase() {}

  void init();
  void close();
  bool process(const cv::Mat &frame, cv::Mat *out);

 protected:
  CodecContext ctx_;
  CodecStartupParams params_;

  virtual void check_valid(const cv::Mat &frame) const {}
};

}  // namespace wdr::media
