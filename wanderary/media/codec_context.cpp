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
                                           int height) {
  DCHECK(codec_id == MediaCodecID::kMJPEG || codec_id == MediaCodecID::kJPEG)
      << "Invalid codec id: " << MediaCodecID2str(codec_id);
  CodecContext ctx;
  auto &c_ctx = ctx.mutable_context();
  c_ctx.codec_id = static_cast<media_codec_id_t>(codec_id);
  c_ctx.encoder = true;

  auto &params = c_ctx.video_enc_params;
  params.width = width;
  params.height = height;
  params.pix_fmt = MC_PIXEL_FORMAT_NV12;
  params.bitstream_buf_size =
      (width * height * 3 / 2 + 0xfff) & ~0xfff;  // align to 4096
  params.frame_buf_count = 5;
  params.external_frame_buf = false;
  params.bitstream_buf_count = 5;

  params.gop_params.decoding_refresh_type = 2;
  params.gop_params.gop_preset_idx = 9;

  params.rot_degree = MC_CCW_0;
  params.mir_direction = MC_DIRECTION_NONE;
  params.frame_cropping_flag = false;
  params.enable_user_pts = 1;

  if (codec_id == MediaCodecID::kMJPEG) {
    params.rc_params.mode = MC_AV_RC_MODE_MJPEGFIXQP;
    const auto stat =
        hb_mm_mc_get_rate_control_config(&c_ctx, &params.rc_params);
    const MediaErrorCode err_code = int2MediaErrorCode(stat);
    CHECK(err_code == MediaErrorCode::kSuccess)
        << "Failed to get rate control config, error code: "
        << MediaErrorCode2str(err_code);

    params.mjpeg_enc_config.restart_interval = width / 16;
  } else {
    LOG(FATAL) << "Unsupported codec: " << MediaCodecID2str(codec_id);
  }

  return ctx;
}

CodecContext CodecContext::CreateJpgDecode(MediaCodecID codec_id, int width,
                                           int height) {
  DCHECK(codec_id == MediaCodecID::kMJPEG || codec_id == MediaCodecID::kJPEG)
      << "Invalid codec id: " << MediaCodecID2str(codec_id);
  CodecContext ctx;
  auto &c_ctx = ctx.mutable_context();

  c_ctx.codec_id = static_cast<media_codec_id_t>(codec_id);
  c_ctx.encoder = false;

  auto &params = c_ctx.video_dec_params;
  params.feed_mode = MC_FEEDING_MODE_FRAME_SIZE;
  params.pix_fmt = MC_PIXEL_FORMAT_NV12;
  params.bitstream_buf_size =
      (width * height * 3 / 2 + 0xfff) & ~0xfff;  // align to 4096
  params.frame_buf_count = 5;
  params.bitstream_buf_count = 5;

  switch (codec_id) {
    case MediaCodecID::kMJPEG:
      params.mjpeg_dec_config.rot_degree = MC_CCW_0;
      params.mjpeg_dec_config.mir_direction = MC_DIRECTION_NONE;
      params.mjpeg_dec_config.frame_crop_enable = false;
      break;
    case MediaCodecID::kJPEG:
      params.jpeg_dec_config.rot_degree = MC_CCW_0;
      params.jpeg_dec_config.mir_direction = MC_DIRECTION_NONE;
      params.jpeg_dec_config.frame_crop_enable = false;
      break;
    default:
      LOG(FATAL) << "Unsupported codec: " << MediaCodecID2str(codec_id);
      break;
  }

  return ctx;
}

int CodecContext::width() const {
  CHECK(is_codec_video(this->id()) && this->encoder());
  return this->ctx_.video_enc_params.width;
}

int CodecContext::height() const {
  CHECK(is_codec_video(this->id()) && this->encoder());
  return this->ctx_.video_enc_params.height;
}

}  // namespace wdr::media
