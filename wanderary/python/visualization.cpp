#include <string>
#include <vector>

#include "wanderary/python/wdr.h"
#include "wanderary/visualization/draw_boxes.h"

using Box2DDrawer = wdr::vis::Box2DDrawer;

void BindBox2DDrawer(py::module *m) {
  py::class_<Box2DDrawer> draw2d_class(*m, "Box2DDrawer");
  draw2d_class.def(py::init<int, const std::vector<std::string> &>(),
                   py::arg("class_num"),
                   py::arg("class_names") = std::vector<std::string>());
  draw2d_class.def("draw",
                   [](Box2DDrawer *self, const std::vector<wdr::Box2D> &boxes,
                      const py::array_t<uint8_t> &img) {
                     cv::Mat img_cv = PyArray2CvMat<uint8_t>(img);
                     self->draw(boxes, &img_cv);
                     return CvMat2PyArray<uint8_t>(img_cv);
                   });
}

void BindVisualization(py::module *m) { BindBox2DDrawer(m); }
