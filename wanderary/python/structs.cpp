#include "wanderary/python/wdr.h"
#include "wanderary/structs/box.h"

using Box2D = wdr::Box2D;

void BindBox2DLabel(py::module *m) {
  py::class_<Box2D::Label> label_class(*m, "Label2D");
  label_class.def(py::init<>());
  label_class.def_readwrite("id", &Box2D::Label::id);
}

void BindBox2D(py::module *m) {
  py::class_<Box2D> box2d_class(*m, "Box2D");
  box2d_class.def(py::init<>());
  box2d_class.def_readwrite("score", &Box2D::score);
  box2d_class.def_readwrite("x_min", &Box2D::x_min);
  box2d_class.def_readwrite("y_min", &Box2D::y_min);
  box2d_class.def_readwrite("h", &Box2D::h);
  box2d_class.def_readwrite("w", &Box2D::w);
  box2d_class.def_readwrite("label", &Box2D::label);

  box2d_class.def("dump", &Box2D::dump);
  box2d_class.def_static("load", &Box2D::load, py::arg("data"));
  box2d_class.def("__str__",
                  [](const Box2D &box) { return box.dump().dump(-1); });
  box2d_class.def("__repr__",
                  [](const Box2D &box) { return box.dump().dump(-1); });
  box2d_class.def("__copy__", [](const Box2D &box) { return box; });
  box2d_class.def("__deepcopy__",
                  [](const Box2D &box, py::dict) { return box; });
}

void BindStructs(py::module *m) {
  BindBox2DLabel(m);
  BindBox2D(m);
}
