#include "wanderary/dnn/bpu_nets.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace wdr::dnn {

ModelData::ModelData(const hbDNNHandle_t &handle) {
  this->dnn_handle_ = handle;
  createTensors(handle, /*fetch_input = */ true, &this->input_tensors_);
  createTensors(handle, /*fetch_input = */ false, &this->output_tensors_);
}

ModelData::~ModelData() {
  releaseTensors(true, &this->input_tensors_);
  releaseTensors(false, &this->output_tensors_);
}

BPUNets::BPUNets(const std::vector<std::string> &modelpaths) {
  // 加载模型/构造所有模型的handles/输入输出Tensors
  std::map<std::string, hbDNNHandle_t> handles;
  readNets(modelpaths, &packed_handle_, &handles);
  net_map_.clear();
  for (const auto &[name, handle] : handles) {
    net_map_.emplace(name, ModelData(handle));
    LOG(INFO) << "Loaded BPU model: " << name;
  }
}

BPUNets::~BPUNets() {
  // Tensor会由ModelData负责释放
  releaseNets(&this->packed_handle_);
}

}  // namespace wdr::dnn
