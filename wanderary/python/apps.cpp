#include <string>
#include <vector>

#include "wanderary/apps/yolo.h"
#include "wanderary/python/wdr.h"

using YOLOv8 = wdr::apps::YOLOv8;
using StatisticsTimeManager = wdr::StatisticsTimeManager;

void BindYolo(py::module *m) {
  py::class_<YOLOv8> yolo_class(*m, "YOLOv8");
  yolo_class.def(py::init<const std::string &, const wdr::json &, int>(),
                 py::arg("appname"), py::arg("cfg"), py::arg("thread_num"));

  yolo_class.def(
      "run",
      [](YOLOv8 *self, const py::array_t<uint8_t> &img,
         const std::vector<py::tuple> &rois, StatisticsTimeManager *stats) {
        const cv::Mat img_cv = PyArray2CvMat<uint8_t>(img);
        const int roi_num = rois.size();
        std::vector<cv::Rect> rois_cv(roi_num);
        for (int i = 0; i < roi_num; i++) rois_cv[i] = PyTuple2Rect(rois[i]);

        std::vector<wdr::Box2D> boxes = self->run(img_cv, rois_cv, stats);
        return boxes;
      },
      py::arg("img"), py::arg("rois"), py::arg("stats") = nullptr);
}

void BindApps(py::module *m) { BindYolo(m); }
