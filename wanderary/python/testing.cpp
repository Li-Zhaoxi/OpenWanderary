#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "wanderary/python/wdr.h"
#include "wanderary/testing/data_checking.h"

void BindTesting(py::module *m) {
  m->def(
      "CheckUnorderedBoxes",
      [](const std::vector<wdr::Box2D> &pred, const std::vector<wdr::Box2D> &gt,
         double eps, const std::string &msg) {
        wdr::testing::UnorderedCheck(pred, gt, eps, msg);
        return !::testing::Test::HasFailure();
      },
      py::arg("pred"), py::arg("gt"), py::arg("eps"), py::arg("msg") = "");
}
