#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <wanderary/dnn/bpu_core.h>

#include <opencv2/opencv.hpp>

namespace wdr::dnn {

struct ModelData {
  explicit ModelData(const hbDNNHandle_t &handle);
  ~ModelData();
  hbDNNHandle_t dnn_handle_{nullptr};
  std::vector<hbDNNTensor> input_tensors_;
  std::vector<hbDNNTensor> output_tensors_;
};

class BPUNets {
 public:
  explicit BPUNets(const std::vector<std::string> &modelpaths);
  ~BPUNets();

  void Forward(const std::string &model_name,
               const std::vector<cv::Mat> &input_mats,
               std::vector<cv::Mat> *out_feats);

 private:
  ModelData *GetModelData(const std::string &model_name);
  hbPackedDNNHandle_t packed_handle_{nullptr};
  std::map<std::string, std::unique_ptr<ModelData>> net_map_;
};

}  // namespace wdr::dnn
