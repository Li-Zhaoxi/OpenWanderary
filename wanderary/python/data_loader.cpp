#include "wanderary/data_loader/data_loader.h"

#include "wanderary/python/wdr.h"

using DataLoader = wdr::loader::DataLoader;

void BindLoader(py::module *m) {
  py::class_<DataLoader> loader_class(*m, "DataLoader");
  loader_class.def(py::init<const wdr::json &>(), py::arg("config"));
  loader_class.def("size", &DataLoader::size);

  loader_class.def_static("RegisteredNames", &DataLoader::RegisteredNames);
}

void BindDataLoader(py::module *m) { BindLoader(m); }
