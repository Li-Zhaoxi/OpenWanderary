#include <memory>
#include <optional>

#include <glog/logging.h>

#include "wanderary/media/codec_base.h"
#include "wanderary/media/codec_jpg.h"
#include "wanderary/media/media_codec.h"
#include "wanderary/python/wdr.h"

using MediaCodecBase = wdr::media::MediaCodecBase;
using MediaCodecJpg = wdr::media::MediaCodecJpg;
using CodecContext = wdr::media::CodecContext;

void BindMediaCodecID(py::module *m) {
  py::enum_<MediaCodecID> enum_type(*m, "MediaCodecID", py::arithmetic());
  for (const auto &[val, strval] : map_MediaCodecID2strvar)
    enum_type.value(strval.c_str(), val);
}

class PyMediaCodecBase : MediaCodecBase {
 public:
  using MediaCodecBase::MediaCodecBase;

 protected:
  void check_valid(const cv::Mat &frame) const override {
    PYBIND11_OVERLOAD_PURE(void, MediaCodecBase, check_valid, frame);
  }

  void prepare_dequeue_input(CodecContext *ctx,
                             media_codec_buffer_t *buf) const override {
    PYBIND11_OVERLOAD_PURE(void, MediaCodecBase, prepare_dequeue_input, ctx,
                           buf);
  }
  void prepare_queue_input(const cv::Mat &frame, CodecContext *ctx,
                           media_codec_buffer_t *buf) const override {
    PYBIND11_OVERLOAD_PURE(void, MediaCodecBase, prepare_queue_input, frame,
                           ctx, buf);
  }

  void prepare_output(const media_codec_buffer_t &buf, const CodecContext &ctx,
                      const media_codec_output_buffer_info_t &info,
                      cv::Mat *out) const override {
    PYBIND11_OVERLOAD_PURE(void, MediaCodecBase, prepare_output, buf, ctx, info,
                           out);
  }
};

void BindMediaCodecBase(py::module *m) {
  py::class_<MediaCodecBase, std::shared_ptr<MediaCodecBase>, PyMediaCodecBase>
      codec_class(*m, "MediaCodecBase");
  codec_class.def(py::init<MediaCodecID, bool>(), py::arg("codec_id"),
                  py::arg("encode"));
  codec_class.def("init", &MediaCodecBase::init);
  codec_class.def("close", &MediaCodecBase::close);
  codec_class.def(
      "process",
      [](MediaCodecBase *self, const py::array_t<uint8_t> &frame)
          -> std::optional<py::array_t<uint8_t>> {
        cv::Mat_<uint8_t> cvframe = PyArray2CvMat(frame);
        cv::Mat out;
        const bool res = self->process(cvframe, &out);
        if (!res) return std::nullopt;
        return std::optional<py::array_t<uint8_t>>(CvMat2PyArray<uint8_t>(out));
      },
      py::arg("frame"));
}

void BindMediaCodecJpg(py::module *m) {
  py::class_<MediaCodecJpg, std::shared_ptr<MediaCodecJpg>, MediaCodecBase>
      codec_class(*m, "MediaCodecJpg");
  codec_class.def(py::init<MediaCodecID, bool, int, int>(), py::arg("codec_id"),
                  py::arg("encode"), py::arg("width"), py::arg("height"));
}

void BindMedia(py::module *m) {
  BindMediaCodecID(m);
  BindMediaCodecBase(m);
  BindMediaCodecJpg(m);
}
