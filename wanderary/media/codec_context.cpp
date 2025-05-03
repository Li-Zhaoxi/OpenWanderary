#include <string>

#include <glog/logging.h>

#include "wanderary/media/media_codec.h"
#include "wanderary/media/media_error.h"

namespace wdr::media {

CodecContext::CodecContext() { memset(&this->ctx_, 0x00, sizeof(this->ctx_)); }

MediaCodecID CodecContext::id() const {
  return int2MediaCodecID(this->ctx_.codec_id);
}

CodecContext CodecContext::CreateJpgEncode(MediaCodecID codec_id, int width,
                                           int height,
                                           CodecPixelFormat pixfmt) {
  DCHECK(codec_id == MediaCodecID::kMJPEG || codec_id == MediaCodecID::kJPEG)
      << "Invalid codec id: " << MediaCodecID2str(codec_id);
  CodecContext ctx;
  auto &c_ctx = ctx.mutable_context();
  c_ctx.codec_id = static_cast<media_codec_id_t>(codec_id);
  c_ctx.encoder = true;

  auto &params = c_ctx.video_enc_params;
  params.width = width;
  params.height = height;
  params.pix_fmt = static_cast<mc_pixel_format_t>(pixfmt);
  params.frame_buf_count = 5;
  params.bitstream_buf_count = 5;
  params.rot_degree = MC_CCW_0;
  params.mir_direction = MC_DIRECTION_NONE;
  params.frame_cropping_flag = false;
  params.external_frame_buf = false;

  if (codec_id == MediaCodecID::kMJPEG) {
    params.rc_params.mode = MC_AV_RC_MODE_MJPEGFIXQP;
    const auto stat =
        hb_mm_mc_get_rate_control_config(&c_ctx, &params.rc_params);
    const MediaErrorCode err_code = int2MediaErrorCode(stat);
    CHECK(err_code == MediaErrorCode::kSuccess)
        << "Failed to get rate control config, error code: "
        << MediaErrorCode2str(err_code);
    params.mjpeg_enc_config.restart_interval = width / 16;
    params.mjpeg_enc_config.extended_sequential = true;
  } else {
    params.jpeg_enc_config.restart_interval = width / 16;
    params.jpeg_enc_config.extended_sequential = true;
  }

  return ctx;
}

}  // namespace wdr::media
