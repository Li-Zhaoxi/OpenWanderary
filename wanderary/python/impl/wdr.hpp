#pragma once

#include <glog/logging.h>
#include <wanderary/python/wdr.h>

template <typename DType>
cv::Mat PyArray2CvMat(const py::array_t<DType> &pydata) {
  py::buffer_info buf = pydata.request();

  if (buf.ndim != 2 && buf.ndim != 3) {
    std::stringstream ss;
    ss << "The input array must be 2D/3D, but it is " << buf.ndim << "D.";
    py::set_error(PyExc_ValueError, ss.str().c_str());
  }
  const int rows = buf.shape[0];
  const int cols = buf.shape[1];
  const int chls = buf.ndim == 3 ? buf.shape[2] : 1;

  cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<DType>::type, chls));
  const int bytesize = mat.total() * mat.elemSize();

  if (bytesize != buf.itemsize * buf.size) {
    std::stringstream ss;
    ss << "The input array size is not correct. Expected " << bytesize
       << " bytes, but got " << buf.itemsize * buf.size << " bytes.";
    py::set_error(PyExc_ValueError, ss.str().c_str());
  }
  if (!(pydata.flags() & py::array::c_style)) {
    py::array_t<DType, py::array::c_style> tmpdata = pydata;
    memcpy(mat.data, tmpdata.data(), bytesize);

  } else {
    memcpy(mat.data, buf.ptr, bytesize);
  }
  return mat;
}

template <typename DType>
py::array_t<DType> CvMat2PyArray(const cv::Mat &cvdata) {
  if (cvdata.depth() != cv::DataType<DType>::type) {
    std::stringstream ss;
    ss << "The input Mat type is " << cvdata.type() << ", not "
       << cv::DataType<DType>::type << ".";
    py::set_error(PyExc_ValueError, ss.str().c_str());
  }
  const int rows = cvdata.rows;
  const int cols = cvdata.cols;
  const int chls = cvdata.channels();

  if (rows <= 0 || cols <= 0 || chls <= 0) {
    std::stringstream ss;
    ss << "The input Mat size is not correct. Got " << rows << "x" << cols
       << "x" << chls << ".";
    py::set_error(PyExc_ValueError, ss.str().c_str());
  }

  py::array_t<DType> pydata;
  if (chls == 1)
    pydata = py::array_t<DType, py::array::c_style>({rows, cols});
  else
    pydata = py::array_t<DType, py::array::c_style>({rows, cols, chls});
  py::buffer_info buf = pydata.request();
  const int bytesize = cvdata.total() * cvdata.elemSize();
  if (buf.size != bytesize) {
    std::stringstream ss;
    ss << "The input Mat size is not correct. Got " << buf.size
       << " bytes, but expected " << bytesize << " bytes.";
    py::set_error(PyExc_ValueError, ss.str().c_str());
  }
  if (cvdata.isContinuous()) {
    memcpy(buf.ptr, cvdata.data, bytesize);
  } else {
    cv::Mat tmp;
    cvdata.copyTo(tmp);
    memcpy(buf.ptr, tmp.data, bytesize);
  }
  return pydata;
}
