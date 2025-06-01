#include "wanderary/media/codec_jpg.h"

#include <glog/logging.h>

namespace wdr::media {

MediaCodecJpg::MediaCodecJpg(MediaCodecID codec_id, bool encode, int width,
                             int height)
    : nv12_size_(width * height * 3 / 2), MediaCodecBase(codec_id, encode) {
  if (encode)
    this->ctx_ = CodecContext::CreateJpgEncode(codec_id, width, height,
                                               CodecPixelFormat::kNV12);
  else
    LOG(FATAL) << "Jpg decode not supported yet";
}

void MediaCodecJpg::check_valid(const cv::Mat &frame) const {
  if (this->ctx_.encoder()) {
    CHECK_EQ(frame.rows, 1);
    CHECK_EQ(frame.cols, this->nv12_size_);
    CHECK_EQ(frame.channels(), 1);
    CHECK_EQ(frame.type(), CV_8UC1);
  } else {
    LOG(FATAL) << "Jpg decode not supported yet";
  }
}

void MediaCodecJpg::prepare_dequeue_input(CodecContext *ctx,
                                          media_codec_buffer_t *buf) const {
  buf->type = ctx->encoder() ? MC_VIDEO_FRAME_BUFFER : MC_VIDEO_STREAM_BUFFER;
}

void MediaCodecJpg::prepare_queue_input(const cv::Mat &frame, CodecContext *ctx,
                                        media_codec_buffer_t *buf) const {
  if (ctx->encoder()) {
    const auto &c_ctx = ctx->context();
    const int width = c_ctx.video_enc_params.width;
    const int height = c_ctx.video_enc_params.height;
    buf->type = MC_VIDEO_FRAME_BUFFER;
    buf->vframe_buf.width = width;
    buf->vframe_buf.height = height;
    buf->vframe_buf.pix_fmt = MC_PIXEL_FORMAT_NV12;
    buf->vframe_buf.size = width * height * 3 / 2;
    buf->vframe_buf.pts = 0;
    memcpy(buf->vframe_buf.vir_ptr[0], frame.data, nv12_size_);
  } else {
    CHECK_GE(buf->vstream_buf.size, nv12_size_);
    buf->type = MC_VIDEO_STREAM_BUFFER;
    LOG(FATAL) << "not implemented";
  }
}

void MediaCodecJpg::prepare_output(const media_codec_buffer_t &buf,
                                   const CodecContext &ctx,
                                   cv::Mat *out) const {
  if (!ctx.encoder() && (buf.type != MC_VIDEO_FRAME_BUFFER)) {
    LOG(FATAL) << "decoder not work";
  }

  if (is_codec_video(ctx.id())) {
    if (ctx.encoder()) {
      const int data_size = buf.vstream_buf.size;
      LOG(INFO) << "encode data_size: " << data_size;
      out->create(1, data_size, CV_8UC1);
      memcpy(out->data, buf.vstream_buf.vir_ptr, data_size);
    } else {
      const int w = buf.vframe_buf.width;
      const int h = buf.vframe_buf.height;
      LOG(FATAL) << "not support";
    }
  }
}

}  // namespace wdr::media
