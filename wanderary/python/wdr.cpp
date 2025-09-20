#include "wanderary/python/wdr.h"

#include <string>

cv::Mat PyObject2CvMat(const py::object &obj) {
  cv::Mat res;
  if (py::isinstance<py::array_t<float>>(obj))
    res = PyArray2CvMat<float>(obj.cast<py::array_t<float>>());
  else if (py::isinstance<py::array_t<uint8_t>>(obj))
    res = PyArray2CvMat<uint8_t>(obj.cast<py::array_t<uint8_t>>());
  else if (py::isinstance<py::array_t<int32_t>>(obj))
    res = PyArray2CvMat<int32_t>(obj.cast<py::array_t<int32_t>>());
  else
    LOG(FATAL) << "Unsupported type: "
               << obj.attr("__name__").cast<std::string>();
  return res;
}

py::object CvMat2PyObject(const cv::Mat &cvdata) {
  py::object res;
  if (cvdata.depth() == CV_8U)
    res = CvMat2PyArray<uint8_t>(cvdata);
  else if (cvdata.depth() == CV_32F)
    res = CvMat2PyArray<float>(cvdata);
  else
    LOG(FATAL) << "Unsupported type: " << cvdata.depth();
  return res;
}

cv::Rect PyTuple2Rect(const py::tuple &t) {
  CHECK_EQ(t.size(), 4);
  return cv::Rect(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(),
                  t[3].cast<int>());
}
