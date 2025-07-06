#include "wanderary/python/wdr.h"

PYBIND11_MODULE(wanderary, m) {
  m.doc() = "wanderary";
  BindUtils(&m);
  BindMedia(&m);
  BindProcess(&m);
  BindDNN(&m);
}
