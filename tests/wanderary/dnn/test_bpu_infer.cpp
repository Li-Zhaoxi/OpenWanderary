#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/dnn/bpu_core.h"
#include "wanderary/testing/data_checking.h"

TEST(DNNBPU, PrintInfoByAPIs) {
  const std::string binpath =
      "../../test_data/models/yolov8n_detect_bayese_640x640_nv12_modified.bin";
  const std::string gapth =
      "../../test_data/dnn/yolov8n_detect_bayese_640x640_nv12_properties.json";
  // 加载模型
  hbPackedDNNHandle_t pPackedNets = nullptr;
  std::map<std::string, hbDNNHandle_t> netsMap;
  wdr::dnn::readNets({binpath}, &pPackedNets, &netsMap);

  const auto gtprop = wdr::LoadJson(gapth);

  // 打印模型个数，以及模型名称
  const int model_num = netsMap.size();
  EXPECT_EQ(model_num, 1);
  int model_count = 0;
  for (const auto &net : netsMap) {
    LOG(INFO) << "[" << model_count << "/" << model_num
              << "] model name: " << net.first;
    model_count++;
  }
  if (model_num != 1) return;
  EXPECT_EQ(netsMap.begin()->first, "yolov8n_detect_bayese_640x640_nv12");

  // 获取第一个模型的输入输出属性
  const hbDNNHandle_t handle_model = netsMap.begin()->second;
  std::vector<hbDNNTensorProperties> input_properties;
  std::vector<hbDNNTensorProperties> output_properties;
  wdr::dnn::readNetProperties(handle_model, true, &input_properties);
  wdr::dnn::readNetProperties(handle_model, false, &output_properties);

  // 整合Tensor属性
  wdr::json input_data;
  wdr::json output_data;
  for (const auto &prop : input_properties)
    input_data.push_back(wdr::dnn::dump(prop));
  for (const auto &prop : output_properties)
    output_data.push_back(wdr::dnn::dump(prop));
  wdr::json res;
  res["input"] = std::move(input_data);
  res["output"] = std::move(output_data);
  LOG(INFO) << "model properties: " << res.dump(2);

  wdr::SaveJson("model_properties.json", res);
  wdr::testing::Check(res, gtprop, 1e-6, "PrintInfoByAPIs");

  // 释放网络
  wdr::dnn::releaseNets(&pPackedNets);
}
