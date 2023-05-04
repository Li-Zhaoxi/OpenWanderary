// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _METHOD_PTQ_FCOS_POST_PROCESS_METHOD_H_
#define _METHOD_PTQ_FCOS_POST_PROCESS_METHOD_H_

#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

/**
 * Config definition for Fcos
 */
struct PTQFcosConfig {
  std::vector<int> strides;
  int class_num;
  std::vector<std::string> class_names;
  std::string det_name_list;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }
    ss << "; class_num: " << class_num;
    return ss.str();
  }
};

extern PTQFcosConfig default_ptq_fcos_config;
/**
 * Method for post processing
 */
class PTQFcosPostProcessMethod : public PostProcessMethod {
 public:
  /**
   * Init post process from json string
   * @param[in] config: config json string
   *        config file should be in the json format
   *        for example:
   *        {
   *          "top_k": 1,
   *          "cls_names_list":
   * "../../config/model/data_name_list/imagenet.list"
   *        }
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~PTQFcosPostProcessMethod() override = default;

private:
  void GetBboxAndScoresNoneNCHW(std::vector<hbDNNTensor> &tensors,
                                ImageTensor *image_tensor,
                                std::vector<Detection> &dets);
  void GetBboxAndScoresNoneNHWC(std::vector<hbDNNTensor> &tensors,
                                ImageTensor *image_tensor,
                                std::vector<Detection> &dets);
  void GetBboxAndScoresScaleNHWC(std::vector<hbDNNTensor> &tensors,
                                 ImageTensor *image_tensor,
                                 std::vector<Detection> &dets);
  void CqatGetBboxAndScoresScaleNHWC(std::vector<hbDNNTensor> &tensors,
                                     ImageTensor *image_tensor,
                                     std::vector<Detection> &dets);
  int PostProcess(std::vector<hbDNNTensor> &tensors,
                  ImageTensor *image_tensor,
                  Perception *perception);

 private:
  PTQFcosConfig fcos_config_ = default_ptq_fcos_config;
  int topk_ = 100;
  float score_threshold_ = 0.1;
  std::ifstream ifs_;
  bool community_qat_ = false;
};

#endif  // _METHOD_PTQ_FCOS_POST_PROCESS_METHOD_H_
