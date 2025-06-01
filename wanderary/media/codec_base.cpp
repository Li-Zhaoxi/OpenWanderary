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

bool MediaCodecBase::process(const cv::Mat &frame, cv::Mat *out) {
  MediaErrorCode err_code = MediaErrorCode::kUnknown;

  // 状态校验
  const auto state = wdr::media::GetCodecState(&ctx_);
  CHECK(state == MediaCodecState::kStarted)
      << "codec_state: " << MediaCodecState2str(state);

  this->check_valid(frame);

  // 退出input_buffer
  this->prepare_dequeue_input(&ctx_, &buf_);
  err_code = int2MediaErrorCode(
      hb_mm_mc_dequeue_input_buffer(&ctx_.mutable_context(), &buf_, 100));
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "dequeue_input_buffer failed, err_code: "
      << MediaErrorCode2str(err_code);

  // 准备input_buffer
  this->prepare_queue_input(frame, &ctx_, &buf_);
  err_code = int2MediaErrorCode(
      hb_mm_mc_queue_input_buffer(&ctx_.mutable_context(), &buf_, 2000));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "queue input buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }

  // dequeue出output_buffer
  err_code = int2MediaErrorCode(hb_mm_mc_dequeue_output_buffer(
      &ctx_.mutable_context(), &buf_, &out_info_, 2000));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "dequeue output buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }

  this->prepare_output(buf_, ctx_, out);

  err_code = int2MediaErrorCode(
      hb_mm_mc_queue_output_buffer(&ctx_.mutable_context(), &buf_, 0));
  CHECK(err_code == MediaErrorCode::kSuccess)
      << "queue output buffer failed, err_code: "
      << MediaErrorCode2str(err_code);

  return true;
}

}  // namespace wdr::media
