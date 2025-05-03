#include "wanderary/media/media_codec.h"

#include <set>

#include "wanderary/media/media_error.h"
namespace wdr::media {

bool is_codec_video(MediaCodecID id) {
  static const std::set<MediaCodecID> video_codecs = {
      MediaCodecID::kH264, MediaCodecID::kH265, MediaCodecID::kMJPEG,
      MediaCodecID::kJPEG};

  return video_codecs.find(id) != video_codecs.end();
}

bool is_codec_jpg(MediaCodecID id) {
  static const std::set<MediaCodecID> jpg_codecs = {MediaCodecID::kJPEG,
                                                    MediaCodecID::kMJPEG};
  return jpg_codecs.find(id) != jpg_codecs.end();
}

CodecStartupParams::CodecStartupParams(MediaCodecID codec_id, bool encode)
    : codec_id_(codec_id), encode_(encode) {
  memset(&params_, 0x00, sizeof(mc_av_codec_startup_params_t));
}

void CodecStartupParams::set_receive_frame_number(int frame_number) {
  if (is_codec_video(codec_id_)) {
    if (encode_)
      this->params_.video_enc_startup_params.receive_frame_number =
          frame_number;
    else
      LOG(FATAL) << "Video decoder does not support frame number";
  } else {
    LOG(FATAL) << "Unsupported codec: " << MediaCodecID2str(this->codec_id_);
  }
}

CodecDescriptor GetCodecDescriptor(MediaCodecID codec_id) {
  return CodecDescriptor(
      hb_mm_mc_get_descriptor(static_cast<media_codec_id_t>(codec_id)));
}

CodecContext GetDefaultContext(MediaCodecID codec_id, bool encoder) {
  CodecContext res;
  const auto stat = hb_mm_mc_get_default_context(
      static_cast<media_codec_id_t>(codec_id), encoder, &res.mutable_context());

  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to get default context for codec: "
      << MediaErrorCode2str(err_code);

  return res;
}

void InitializeCodecContext(CodecContext *ctx) {
  const auto stat = hb_mm_mc_initialize(&ctx->mutable_context());
  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to initialize codec context: " << MediaErrorCode2str(err_code);
  LOG(INFO) << "Successfully initialized codec context";
}

MediaCodecState GetCodecState(CodecContext *ctx) {
  media_codec_state_t state;
  const auto res = hb_mm_mc_get_state(&ctx->mutable_context(), &state);
  const MediaErrorCode err_code = int2MediaErrorCode(res);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to get codec state: " << MediaErrorCode2str(err_code);
  return int2MediaCodecState(state);
}

void ReleaseCodecContext(CodecContext *ctx) {
  const auto stat = hb_mm_mc_release(&ctx->mutable_context());
  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to release codec context: " << MediaErrorCode2str(err_code);
  LOG(INFO) << "Successfully released codec context";
}

void CodecConfigure(CodecContext *ctx) {
  const auto stat = hb_mm_mc_configure(&ctx->mutable_context());
  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to configure codec: " << MediaErrorCode2str(err_code);
  LOG(INFO) << "Successfully configured codec";
}

void CodecStart(CodecContext *ctx, CodecStartupParams *params) {
  hb_s32 stat = -1;
  if (params != nullptr) {
    stat = hb_mm_mc_start(&ctx->mutable_context(), &params->mutable_params());
  } else {
    mc_av_codec_startup_params_t params;
    stat = hb_mm_mc_start(&ctx->mutable_context(), &params);
  }
  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to start codec: " << MediaErrorCode2str(err_code);
  LOG(INFO) << "Successfully started codec";
}

void CodecStop(CodecContext *ctx) {
  const auto stat = hb_mm_mc_stop(&ctx->mutable_context());
  const MediaErrorCode err_code = int2MediaErrorCode(stat);
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "Failed to stop codec: " << MediaErrorCode2str(err_code);
  LOG(INFO) << "Successfully stopped codec";
}

}  // namespace wdr::media
