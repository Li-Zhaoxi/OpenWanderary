#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/dnn/bpu_core.h"
#include "wanderary/dnn/bpu_nets.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/time_manager.h"
using TimeManager = wdr::TimerManager;

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

TEST(BPUNets, Forward) {
  const std::vector<std::string> modelpaths = {
      "../../test_data/models/yolov8n_detect_bayese_640x640_nv12_modified.bin"};
  const std::string input_path = "../../test_data/dnn/zidane_input_tensor.bin";
  const std::string out_prefix = "../../test_data/process/yolov8_";
  const std::vector<std::string> out_paths = {
      out_prefix + "out_s_clses.bin", out_prefix + "out_s_bboxes.bin",
      out_prefix + "out_m_clses.bin", out_prefix + "out_m_bboxes.bin",
      out_prefix + "out_l_clses.bin", out_prefix + "out_l_bboxes.bin",
  };
  const std::vector<cv::Vec4i> dims = {
      cv::Vec4i(1, 80, 80, 80), cv::Vec4i(1, 80, 80, 64),
      cv::Vec4i(1, 40, 40, 80), cv::Vec4i(1, 40, 40, 64),
      cv::Vec4i(1, 20, 20, 80), cv::Vec4i(1, 20, 20, 64),
  };

  // 加载依赖的数据
  auto input_data = wdr::ReadBytesFromFile<uchar>(input_path);
  const std::vector<cv::Mat> input_mats = {
      cv::Mat(1, input_data.size(), CV_8UC1, input_data.data())};
  const int output_num = out_paths.size();
  std::vector<std::vector<float>> gt_outs_data(output_num);
  std::vector<cv::Mat> gt_outs(output_num);
  for (int i = 0; i < output_num; ++i) {
    auto &out_data = gt_outs_data[i];
    out_data = wdr::ReadBytesFromFile<float>(out_paths[i]);
    const auto &dim = dims[i];
    int count = dim[0] * dim[1] * dim[2] * dim[3];
    CHECK_EQ(out_data.size(), count);
    // 注意这里输出cls是float，bbox是int, 由于字节数一样因此字节处都用float
    if (i % 2 == 0) {
      gt_outs[i] = cv::Mat(4, dim.val, CV_32FC1, out_data.data());
    } else {
      gt_outs[i] = cv::Mat(4, dim.val, CV_32SC1, out_data.data());
    }
  }

  // 模型初始化
  TimeManager time_manager;

  const std::string model_name = "yolov8n_detect_bayese_640x640_nv12";
  wdr::dnn::BPUNets nets(modelpaths);
  std::vector<cv::Mat> out_feats;
  for (int i = 0; i < 20; i++) {
    std::stringstream ss;
    ss << "forward_" << i;
    time_manager.start(ss.str());
    nets.Forward(model_name, input_mats, &out_feats);
    time_manager.stop(ss.str());
    EXPECT_EQ(out_feats.size(), output_num);
    if (out_feats.size() != output_num) return;
    for (int i = 0; i < output_num; ++i) {
      wdr::testing::CheckNonGeneralMat(out_feats[i], gt_outs[i], 1e-6,
                                       ss.str());
    }
  }
  time_manager.printStatistics();
}
