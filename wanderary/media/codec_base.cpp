#include "wanderary/media/codec_base.h"

namespace wdr::media {

void MediaCodecBase::Init() {
  InitializeCodecContext(&this->ctx_);
  CodecConfigure(&this->ctx_);
  CodecStart(&this->ctx_, &this->params_);
}

void MediaCodecBase::Close() {
  MediaCodecState state = GetCodecState(&this->ctx_);
  if (state != MediaCodecState::kUninitialized) {
    CodecStop(&this->ctx_);
    ReleaseCodecContext(&this->ctx_);
  }
}

}  // namespace wdr::media
