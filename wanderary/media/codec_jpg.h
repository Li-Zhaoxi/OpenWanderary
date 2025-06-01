#pragma once

#include <wanderary/media/codec_base.h>

namespace wdr::media {

class MediaCodecJpg : public MediaCodecBase {
 public:
  MediaCodecJpg(MediaCodecID codec_id, bool encode, int width, int height);
  virtual ~MediaCodecJpg() = default;

 protected:
  /**
   * \brief 检查输入的frame是否符合JPG的编解码要求
   * \note 如果是编码，则要求输入的frame类型为1xN, N就是nv12的bytesize个数
   * \note 如果是解码，目前暂未支持
   */
  void check_valid(const cv::Mat &frame) const override;

  void prepare_dequeue_input(CodecContext *ctx,
                             media_codec_buffer_t *buf) const override;
  void prepare_queue_input(const cv::Mat &frame, CodecContext *ctx,
                           media_codec_buffer_t *buf) const override;
  void prepare_output(const media_codec_buffer_t &buf, const CodecContext &ctx,
                      cv::Mat *out) const override;

 private:
  const int nv12_size_{0};
};

}  // namespace wdr::media
