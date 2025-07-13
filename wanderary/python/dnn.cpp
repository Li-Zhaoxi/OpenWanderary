#include <string>
#include <vector>

#include "wanderary/dnn/bpu_nets.h"
#include "wanderary/python/wdr.h"

using DequantScales = wdr::dnn::DequantScales;
using BPUNets = wdr::dnn::BPUNets;

void BindNets(py::module *m) {
  py::class_<BPUNets> bpu_class(*m, "BPUNets");
  bpu_class.def(py::init<const std::vector<std::string> &>(),
                py::arg("model_paths"));
  bpu_class.def(py::init<const std::string &>(), py::arg("model_path"));
  bpu_class.def(
      "forward",
      [](BPUNets *self, const std::string &model_name, const py::list &pylist) {
        std::vector<cv::Mat> input_mats;
        for (const auto &pyin : pylist) {
          if (py::isinstance<py::array_t<float>>(pyin)) {
            input_mats.push_back(
                PyArray2CvMat<float>(pyin.cast<py::array_t<float>>()));
          } else if (py::isinstance<py::array_t<uint8_t>>(pyin)) {
            input_mats.push_back(
                PyArray2CvMat<uint8_t>(pyin.cast<py::array_t<uint8_t>>()));
          } else {
            std::string type_name = pyin.attr("__name__").cast<std::string>();
            LOG(FATAL) << "Unsupported type: " << type_name;
          }
        }
        std::vector<cv::Mat> out_feats;
        self->Forward(model_name, input_mats, &out_feats);
        py::list pylist_out;
        for (const auto &out_feat : out_feats) {
          if (out_feat.depth() == CV_32F)
            pylist_out.append(CvMat2PyArray<float>(out_feat));
          else if (out_feat.depth() == CV_32S)
            pylist_out.append(CvMat2PyArray<int32_t>(out_feat));
          else if (out_feat.depth() == CV_8U)
            pylist_out.append(CvMat2PyArray<uint8_t>(out_feat));
          else
            LOG(FATAL) << "Unsupported type: " << out_feat.depth();
        }
        return pylist_out;
      },
      py::arg("model_name"), py::arg("input_mats"));
  bpu_class.def("GetDequantScales",
                py::overload_cast<const std::string &>(
                    &BPUNets::GetDequantScales, py::const_),
                py::arg("model_name"));
}

void BindDequantScales(py::module *m) {
  py::class_<DequantScales> parm_class(*m, "DequantScales");
  parm_class.def(py::init<>());
  parm_class.def_readwrite("de_scales", &DequantScales::de_scales);
}

void BindDNN(py::module *m) {
  BindDequantScales(m);
  BindNets(m);
}
