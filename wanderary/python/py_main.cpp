#include "wanderary/python/wdr.h"

PYBIND11_MODULE(wanderary, m) {
  m.doc() = "wanderary";
  BindMedia(&m);
}
