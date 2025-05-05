#include "wanderary/media/codec_base.h"

#include "wanderary/media/media_error.h"

namespace wdr::media {

void MediaCodecBase::init() {
  InitializeCodecContext(&this->ctx_);
  CodecConfigure(&this->ctx_);
  CodecStart(&this->ctx_, &this->params_);
}

void MediaCodecBase::close() {
  MediaCodecState state = GetCodecState(&this->ctx_);
  if (state != MediaCodecState::kUninitialized) {
    CodecStop(&this->ctx_);
    ReleaseCodecContext(&this->ctx_);
  }
}

bool MediaCodecBase::process_input(const cv::Mat &frame) {
  MediaErrorCode err_code = MediaErrorCode::kUnknown;

  const int byte_size = frame.total() * frame.elemSize();

  memset(&buf_, 0, sizeof(buf_));
  err_code = int2MediaErrorCode(
      hb_mm_mc_dequeue_input_buffer(&ctx_.mutable_context(), &buf_, 100));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "dequeue input buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }

  if (ctx_.encoder()) {
    const auto &c_ctx = ctx_.context();
    buf_.type = MC_VIDEO_FRAME_BUFFER;
    buf_.vframe_buf.width = c_ctx.video_enc_params.width;
    buf_.vframe_buf.height = c_ctx.video_enc_params.height;
    buf_.vframe_buf.pix_fmt = c_ctx.video_enc_params.pix_fmt;
    buf_.vframe_buf.size = byte_size;
    memcpy(buf_.vframe_buf.vir_ptr[0], frame.data, byte_size);
  } else {
    CHECK_GE(buf_.vstream_buf.size, byte_size);
    buf_.type = MC_VIDEO_STREAM_BUFFER;
    LOG(FATAL) << "not implemented";
  }

  err_code = int2MediaErrorCode(
      hb_mm_mc_queue_input_buffer(&ctx_.mutable_context(), &buf_, 100));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "queue input buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }
  return true;
}

bool MediaCodecBase::process_output(cv::Mat *out) {
  MediaErrorCode err_code = MediaErrorCode::kUnknown;
  err_code = int2MediaErrorCode(hb_mm_mc_dequeue_output_buffer(
      &ctx_.mutable_context(), &buf_, &out_info_, 100));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "dequeue output buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }

  if (ctx_.encoder() && (buf_.type != MC_VIDEO_FRAME_BUFFER)) {
    LOG(FATAL) << "decoder not work";
  }

  if (is_codec_video(ctx_.id())) {
    if (ctx_.encoder()) {
      const int data_size = buf_.vstream_buf.size;
      out->create(1, data_size, CV_8UC1);
      memcpy(out->data, buf_.vstream_buf.vir_ptr, data_size);
    } else {
      const int w = buf_.vframe_buf.width;
      const int h = buf_.vframe_buf.height;
      LOG(FATAL) << "not support";
    }
  }

  return true;
}

bool MediaCodecBase::process(const cv::Mat &frame, cv::Mat *out) {
  this->check_valid(frame);

  // 状态校验
  const auto state = wdr::media::GetCodecState(&ctx_);
  CHECK(state == MediaCodecState::kStarted)
      << "codec_state: " << MediaCodecState2str(state);

  // 初始化
  MediaErrorCode err_code = MediaErrorCode::kUnknown;

  if (!process_input(frame)) return false;

  if (!process_output(out)) return false;

  return true;
}

}  // namespace wdr::media
