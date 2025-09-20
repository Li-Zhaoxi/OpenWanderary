#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/options.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>

#include <pybind11_json/pybind11_json.hpp>

namespace py = pybind11;

template <typename DType>
cv::Mat PyArray2CvMat(const py::array_t<DType> &pydata);

cv::Mat PyObject2CvMat(const py::object &obj);

template <typename DType>
py::array_t<DType> CvMat2PyArray(const cv::Mat &cvdata);

py::object CvMat2PyObject(const cv::Mat &cvdata);

cv::Rect PyTuple2Rect(const py::tuple &t);

void BindMedia(py::module *m);
void BindUtils(py::module *m);
void BindDataLoader(py::module *m);
void BindProcess(py::module *m);
void BindDNN(py::module *m);
void BindStructs(py::module *m);
void BindTesting(py::module *m);
void BindVisualization(py::module *m);
void BindApps(py::module *m);

#include <wanderary/python/impl/wdr.hpp>
