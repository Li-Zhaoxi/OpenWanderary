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
  media_codec_buffer_t buf_;
  media_codec_output_buffer_info_t out_info_;

  virtual void check_valid(const cv::Mat &frame) const {}

 private:
  bool process_input(const cv::Mat &frame);
  bool process_output(cv::Mat *out);
};

}  // namespace wdr::media
