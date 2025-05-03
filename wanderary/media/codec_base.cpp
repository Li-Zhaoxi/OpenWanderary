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
  this->check_valid(frame);
  MediaErrorCode err_code = MediaErrorCode::kUnknown;

  // 输入准备
  media_codec_buffer_t input_buf;
  memset(&input_buf, 0, sizeof(input_buf));
  err_code = int2MediaErrorCode(
      hb_mm_mc_dequeue_input_buffer(&ctx_.mutable_context(), &input_buf, 3000));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "dequeue input buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }
  memcpy(static_cast<void *>(input_buf.vstream_buf.vir_ptr),
         static_cast<void *>(frame.data), frame.total() * frame.elemSize());
  err_code = int2MediaErrorCode(
      hb_mm_mc_queue_input_buffer(&ctx_.mutable_context(), &input_buf, 100));
  if (err_code != MediaErrorCode::kSuccess) {
    LOG(WARNING) << "queue input buffer failed, err_code: "
                 << MediaErrorCode2str(err_code);
    return false;
  }

  // 输出准备
  media_codec_buffer_t output_buf;
  memset(&output_buf, 0, sizeof(output_buf));
  return true;
}

}  // namespace wdr::media
