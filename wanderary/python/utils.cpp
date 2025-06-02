#include "wanderary/python/wdr.h"
#include "wanderary/utils/convertor.h"

void BindConvertor(py::module *m) {
  m->def("cvtBGR2NV12", [](const py::array_t<uchar> &bgr) {
    cv::Mat nv12;
    wdr::BGRToNV12(PyArray2CvMat<uchar>(bgr), &nv12);
    return CvMat2PyArray<uchar>(nv12);
  });
}

void BindUtils(py::module *m) { BindConvertor(m); }
