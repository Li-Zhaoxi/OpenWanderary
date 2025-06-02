#include "wanderary/python/wdr.h"
#include "wanderary/utils/convertor.h"

void BindConvertor(py::module *m) {
  m->def(
      "cvtBGR2NV12",
      [](const py::array_t<uchar> &bgr) {
        cv::Mat nv12;
        wdr::BGRToNV12(PyArray2CvMat<uchar>(bgr), &nv12);
        return CvMat2PyArray<uchar>(nv12);
      },
      py::arg("bgr"));

  m->def(
      "cvtNV12ToYUV444",
      [](const py::array_t<uchar> &nv12, int width, int height) {
        cv::Mat yuv444;
        wdr::NV12ToYUV444(PyArray2CvMat<uchar>(nv12), cv::Size(width, height),
                          &yuv444);
        return CvMat2PyArray<uchar>(yuv444);
      },
      py::arg("nv12"), py::arg("width"), py::arg("height"));
}

void BindUtils(py::module *m) { BindConvertor(m); }
