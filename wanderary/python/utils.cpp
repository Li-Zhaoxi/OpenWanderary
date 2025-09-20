#include <memory>
#include <string>
#include <vector>

#include "wanderary/python/wdr.h"
#include "wanderary/utils/convertor.h"
#include "wanderary/utils/time_manager.h"

using TimerManager = wdr::TimerManager;
using AutoScopeTimer = wdr::AutoScopeTimer;
using StatisticsTimeManager = wdr::StatisticsTimeManager;

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

  m->def(
      "ImageCropROIs",
      [](const py::tuple &img_size, const py::tuple &crop_size,
         const py::tuple &offset, bool drop_gap) {
        CHECK_EQ(img_size.size(), 2);
        CHECK_EQ(crop_size.size(), 2);
        CHECK_EQ(offset.size(), 2);
        const cv::Size img_size_cv(img_size[0].cast<int>(),
                                   img_size[1].cast<int>());
        const cv::Size crop_size_cv(crop_size[0].cast<int>(),
                                    crop_size[1].cast<int>());
        const cv::Point offset_cv(offset[0].cast<int>(), offset[1].cast<int>());
        const auto rois =
            wdr::ImageCropROIs(img_size_cv, crop_size_cv, offset_cv, drop_gap);
        std::vector<py::tuple> py_rois;
        for (const auto &roi : rois) {
          py_rois.emplace_back(
              py::make_tuple(roi.x, roi.y, roi.width, roi.height));
        }
        return py_rois;
      },
      py::arg("img_size"), py::arg("crop_size"), py::arg("offset"),
      py::arg("drop_gap"));
}

void BindTimeManager(py::module *m) {
  py::class_<TimerManager> timer_class(*m, "TimerManager");
  timer_class.def(py::init<>());
  timer_class.def("start", &TimerManager::start, py::arg("phase"));
  timer_class.def("stop", &TimerManager::stop, py::arg("phase"));
  timer_class.def("reset", &TimerManager::reset);
  timer_class.def("printStatistics",
                  py::overload_cast<const std::string &>(
                      &TimerManager::printStatistics, py::const_),
                  py::arg("phase"));
  timer_class.def(
      "printStatistics",
      py::overload_cast<>(&TimerManager::printStatistics, py::const_));
  timer_class.def("getDuration", &TimerManager::getDuration, py::arg("phase"));

  m->def("GlobalTimerManager", &wdr::GlobalTimerManager,
         py::return_value_policy::reference);
}

class PyAutoScopeTimer {
 public:
  PyAutoScopeTimer(const std::string &phase, TimerManager *manager)
      : phase_(phase), manager_(manager) {}

  void enter() {
    scope_timer_ = std::make_unique<AutoScopeTimer>(phase_, manager_);
  }

  bool leave(py::object exc_type, py::object exc_val, py::object) {
    scope_timer_ = nullptr;
    return !exc_type.is_none();  // 有异常时返回false
  }

 private:
  TimerManager *manager_{nullptr};
  const std::string phase_;
  std::unique_ptr<AutoScopeTimer> scope_timer_;
};

void BindAutoScopeTimer(py::module *m) {
  py::class_<PyAutoScopeTimer> timer_class(*m, "AutoScopeTimer");
  timer_class.def(py::init<const std::string &, TimerManager *>(),
                  py::arg("phase"), py::arg("manager"));
  timer_class.def("__enter__", &PyAutoScopeTimer::enter);
  timer_class.def("__exit__", &PyAutoScopeTimer::leave);
}

void BindStatisticsTimeManager(py::module *m) {
  py::class_<StatisticsTimeManager> timer_class(*m, "StatisticsTimeManager");
  timer_class.def(py::init<>());
  timer_class.def("add", &StatisticsTimeManager::add);
  timer_class.def("printStatistics", &StatisticsTimeManager::printStatistics);
}

void BindUtils(py::module *m) {
  BindConvertor(m);
  BindTimeManager(m);
  BindAutoScopeTimer(m);
  BindStatisticsTimeManager(m);
}
