#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/options.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>

namespace py = pybind11;

template <typename DType>
cv::Mat PyArray2CvMat(const py::array_t<DType> &pydata);

template <typename DType>
py::array_t<DType> CvMat2PyArray(const cv::Mat &cvdata);

void BindMedia(py::module *m);
void BindUtils(py::module *m);

#include <wanderary/python/impl/wdr.hpp>
