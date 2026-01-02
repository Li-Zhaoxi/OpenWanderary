#include <string>

#include "wanderary/io/mcap_writer.h"
#include "wanderary/python/wdr.h"
using MCAPWriter = wdr::io::MCAPWriter;
using ImageFrame = wdr::ImageFrame;

void BindMCapCompression(py::module *m) {
  py::enum_<mcap::Compression> enum_type(*m, "MCAPCompression",
                                         py::arithmetic());
  enum_type.value("None", mcap::Compression::None);
  enum_type.value("Lz4", mcap::Compression::Lz4);
  enum_type.value("Zstd", mcap::Compression::Zstd);
}

void BindMCapWriter(py::module *m) {
  py::class_<MCAPWriter> mcap_class(*m, "MCAPWriter");
  mcap_class.def(
      py::init<const std::string &, uint64_t, mcap::Compression, bool>(),
      py::arg("filepath"), py::arg("chunk_size") = 1024 * 1024,
      py::arg("compression") = mcap::Compression::Zstd,
      py::arg("enable_crcs") = true);
  mcap_class.def("close", &MCAPWriter::close);
  mcap_class.def(
      "WriteImage",
      py::overload_cast<const std::string &, const std::string &, uint32_t>(
          &MCAPWriter::WriteImage),
      py::arg("topic_name"), py::arg("image_path"), py::arg("sequence") = 0);
  mcap_class.def(
      "WriteImage",
      py::overload_cast<const std::string &, const ImageFrame &, uint32_t>(
          &MCAPWriter::WriteImage),
      py::arg("topic_name"), py::arg("frame"), py::arg("sequence") = 0);
  mcap_class.def("WriteImageBox2Ds", &MCAPWriter::WriteImageBox2Ds,
                 py::arg("topic_name"), py::arg("frame"),
                 py::arg("sequence") = 0);
  mcap_class.def("WriteWaymoFrame", &MCAPWriter::WriteWaymoFrame,
                 py::arg("topic_name"), py::arg("bytes"),
                 py::arg("sequence") = 0, py::arg("mmframe") = nullptr);
}

void BindIO(py::module *m) {
  BindMCapCompression(m);
  BindMCapWriter(m);
}
