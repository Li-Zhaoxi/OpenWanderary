#include "wanderary/dnn/bpu_nets.h"

#include <map>
#include <memory>
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
    net_map_.emplace(name, std::make_unique<ModelData>(handle));
    LOG(INFO) << "Loaded BPU model: " << name;
  }
}

BPUNets::~BPUNets() {
  // 先释放Tensors
  for (auto &[name, model_data] : this->net_map_) {
    model_data.reset();
    LOG(INFO) << "Released input/output tensors for model: " << name;
  }
  releaseNets(&this->packed_handle_);
  LOG(INFO) << "Released packed handle";
}

ModelData *BPUNets::GetModelData(const std::string &model_name) {
  auto iter = this->net_map_.find(model_name);
  CHECK(iter != this->net_map_.end()) << "Model not found: " << model_name;
  return iter->second.get();
}

void BPUNets::Forward(const std::string &model_name,
                      const std::vector<cv::Mat> &input_mats,
                      std::vector<cv::Mat> *out_feats) {
  auto model_data = this->GetModelData(model_name);
  CHECK_EQ(input_mats.size(), model_data->input_tensors_.size());
  const int tensor_num = model_data->input_tensors_.size();
  for (int i = 0; i < tensor_num; ++i) {
    bpuMemcpy(input_mats[i], &model_data->input_tensors_[i]);
  }

  // 设置推理控制参数
  hbDNNTaskHandle_t task_handle = NULL;
  hbDNNInferCtrlParam ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param);
  ctrl_param.bpuCoreId = 0;  // 设置核心 ID
  ctrl_param.priority = 0;   // 设置优先级

  // 执行推理
  hbDNNTensor *output = model_data->output_tensors_.data();
  CHECK_EQ(hbDNNInfer(&task_handle, &output, model_data->input_tensors_.data(),
                      model_data->dnn_handle_, &ctrl_param),
           0);

  // 等待任务完成
  CHECK_EQ(hbDNNWaitTaskDone(task_handle, 0), 0);

  // 完成后拷贝数据
  const int output_count = model_data->output_tensors_.size();
  out_feats->resize(output_count);
  for (int i = 0; i < output_count; ++i) {
    bpuMemcpy(&model_data->output_tensors_[i], &out_feats->at(i));
  }

  // 释放任务句柄
  CHECK_EQ(hbDNNReleaseTask(task_handle), 0);
}

}  // namespace wdr::dnn
