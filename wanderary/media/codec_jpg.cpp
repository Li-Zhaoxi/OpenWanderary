#include "wanderary/media/codec_jpg.h"

#include <glog/logging.h>

namespace wdr::media {

MediaCodecJpg::MediaCodecJpg(MediaCodecID codec_id, bool encode, int width,
                             int height, CodecPixelFormat pixfmt) {
  if (encode)
    this->ctx_ = CodecContext::CreateJpgEncode(codec_id, width, height, pixfmt);
  else
    LOG(FATAL) << "Jpg decode not supported yet";
}

}  // namespace wdr::media
