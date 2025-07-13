#include <memory>
#include <string>
#include <vector>

#include "wanderary/process/process_base.h"
#include "wanderary/python/wdr.h"

using ImageAffineParms = wdr::proc::ImageAffineParms;
using ProcessManager = wdr::proc::ProcessManager;
using ProcessRecorder = wdr::proc::ProcessRecorder;
using ProcessBase = wdr::proc::ProcessBase;

void BindImageAffineParms(py::module *m) {
  py::class_<ImageAffineParms> parm_class(*m, "ImageAffineParms");
  parm_class.def(py::init<>());
  parm_class.def_readwrite("x_scale", &ImageAffineParms::x_scale);
  parm_class.def_readwrite("y_scale", &ImageAffineParms::y_scale);
  parm_class.def_readwrite("x_shift", &ImageAffineParms::x_shift);
  parm_class.def_readwrite("y_shift", &ImageAffineParms::y_shift);
}

void BindProcessRecorder(py::module *m) {
  py::class_<ProcessRecorder> record_class(*m, "ProcessRecorder");
  record_class.def(py::init<>());
  record_class.def_readwrite("affine", &ProcessRecorder::affine);
  record_class.def_readwrite("dequant_scales",
                             &ProcessRecorder::dequant_scales);
}

void BindProcessManager(py::module *m) {
  py::class_<ProcessManager> proc_class(*m, "ProcessManager");
  proc_class.def(py::init<const wdr::json &>(), py::arg("config"));
  proc_class.def(
      "Forward",
      [](ProcessManager *self, const py::array_t<uchar> &pydata,
         ProcessRecorder *recorder) -> py::array_t<uchar> {
        cv::Mat data = PyArray2CvMat<uchar>(pydata);
        self->Forward(&data, recorder);
        return CvMat2PyArray<uchar>(data);
      },
      py::arg("data"), py::arg("recorder") = nullptr);

  proc_class.def(
      "Forward2D",
      [](ProcessManager *self, const py::list &pydata,
         ProcessRecorder *recorder) -> std::vector<wdr::Box2D> {
        std::vector<cv::Mat> data;
        for (const auto &item : pydata)
          data.push_back(PyObject2CvMat(item.cast<py::object>()));
        std::vector<wdr::Box2D> box2ds;
        self->Forward(&data, &box2ds, recorder);
        return box2ds;
      },
      py::arg("data"), py::arg("recorder") = nullptr);

  proc_class.def_static("RegisteredNames", &ProcessManager::RegisteredNames);
}

void BindProcess(py::module *m) {
  BindImageAffineParms(m);
  BindProcessRecorder(m);
  BindProcessManager(m);
}
