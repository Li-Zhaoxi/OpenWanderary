#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/process/process_base.h"
#include "wanderary/process/processors/convert_yolo_feature.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/testing/data_convertor.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/time_manager.h"

using ProcessRecorder = wdr::proc::ProcessRecorder;
using TimeManager = wdr::TimerManager;
using ConvertYoloFeature = wdr::proc::ConvertYoloFeature;
using ImageAffineParms = wdr::proc::ImageAffineParms;
using DequantScales = wdr::proc::DequantScales;

TEST(ProcessBase, TestConvertYoloFeature) {
  TimeManager time_manager;
  // 初始化处理器信息
  const std::string cfg = R"({
    "class_num": 80,
    "reg_num": 16,
    "nms_thres": 0.7,
    "score_thres": 0.25
  })";
  ConvertYoloFeature proc(wdr::json::parse(cfg));

  // 构造辅助信息
  const std::string dataroot = "../../test_data/process/yolov8_";

  ProcessRecorder recorder;
  recorder.affine = ImageAffineParms(0.5, 0.5, 0, 140);
  DequantScales de_scales;
  de_scales.box_scales = {{1, 8}, {3, 16}, {5, 32}};
  de_scales.de_scales = {
      {1, wdr::ReadBytesFromFile<float>(dataroot + "s_bboxes_scale.bin")},
      {3, wdr::ReadBytesFromFile<float>(dataroot + "m_bboxes_scale.bin")},
      {5, wdr::ReadBytesFromFile<float>(dataroot + "l_bboxes_scale.bin")}};
  recorder.dequant_scales = std::move(de_scales);

  // 构造输入数据
  auto s_scls = wdr::ReadBytesFromFile<float>(dataroot + "out_s_clses.bin");
  CHECK_EQ(s_scls.size(), 80 * 80 * 80);
  auto s_boxes = wdr::ReadBytesFromFile<float>(dataroot + "out_s_bboxes.bin");
  CHECK_EQ(s_boxes.size(), 80 * 80 * 64);
  auto m_scls = wdr::ReadBytesFromFile<float>(dataroot + "out_m_clses.bin");
  CHECK_EQ(m_scls.size(), 40 * 40 * 80);
  auto m_bboxes = wdr::ReadBytesFromFile<float>(dataroot + "out_m_bboxes.bin");
  CHECK_EQ(m_bboxes.size(), 40 * 40 * 64);
  auto l_scls = wdr::ReadBytesFromFile<float>(dataroot + "out_l_clses.bin");
  CHECK_EQ(l_scls.size(), 20 * 20 * 80);
  auto l_bboxes = wdr::ReadBytesFromFile<float>(dataroot + "out_l_bboxes.bin");
  CHECK_EQ(l_bboxes.size(), 20 * 20 * 64);
  const std::vector<cv::Mat> feats = {
      cv::Mat({1, 80, 80, 80}, CV_32FC1, s_scls.data()),
      cv::Mat({1, 80, 80, 64}, CV_32SC1, s_boxes.data()),
      cv::Mat({1, 40, 40, 80}, CV_32FC1, m_scls.data()),
      cv::Mat({1, 40, 40, 64}, CV_32SC1, m_bboxes.data()),
      cv::Mat({1, 20, 20, 80}, CV_32FC1, l_scls.data()),
      cv::Mat({1, 20, 20, 64}, CV_32SC1, l_bboxes.data())};

  // 执行推理
  std::vector<wdr::Box2D> box2ds;
  for (int i = 0; i < 10; i++) {
    time_manager.reset();
    time_manager.start("ConvertYoloFeature");
    proc.Forward(feats, &box2ds, &recorder);
    time_manager.stop("ConvertYoloFeature");
    time_manager.printStatistics();
  }
  for (const auto &box2d : box2ds) LOG(INFO) << box2d.dump().dump();

  // 构造GT
  std::vector<wdr::Box2D> gt_box2ds;
  const auto gtdata =
      wdr::LoadJson("../../test_data/process/yolov8_gt_box2ds.json");
  for (const auto &item : gtdata) gt_box2ds.push_back(wdr::Box2D::load(item));

  wdr::testing::UnorderedCheck(box2ds, gt_box2ds, 1e-4);
}
