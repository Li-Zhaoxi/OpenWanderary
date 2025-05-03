#include "wanderary/media/codec_jpg.h"

#include <glog/logging.h>

namespace wdr::media {

MediaCodecJpg::MediaCodecJpg(MediaCodecID codec_id, bool encode, int width,
                             int height, CodecPixelFormat pixfmt)
    : MediaCodecBase(codec_id, encode) {
  if (encode)
    this->ctx_ = CodecContext::CreateJpgEncode(codec_id, width, height, pixfmt);
  else
    LOG(FATAL) << "Jpg decode not supported yet";
}

void MediaCodecJpg::check_valid(const cv::Mat &frame) const {
  if (this->ctx_.encoder()) {
    CHECK_EQ(frame.rows, ctx_.height());
    CHECK_EQ(frame.cols, ctx_.width());
  }
}

}  // namespace wdr::media
