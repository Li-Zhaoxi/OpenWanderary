#pragma once
#include <string>
#include <vector>

#include <tbb/tbb.h>
#include <wanderary/dnn/bpu_nets.h>
#include <wanderary/process/process_base.h>
#include <wanderary/structs/box.h>
#include <wanderary/utils/json_utils.h>
#include <wanderary/utils/time_manager.h>

namespace wdr::apps {

using ProcessManager = wdr::proc::ProcessManager;
using BPUNets = wdr::dnn::BPUNets;
using TimerManager = wdr::TimerManager;
using StatisticsTimeManager = wdr::StatisticsTimeManager;

class BaseYOLOv8 {
 public:
  BaseYOLOv8(const std::string &appname, const wdr::json &cfg);

  std::vector<wdr::Box2D> run(cv::Mat *image,
                              TimerManager *time_manager = nullptr,
                              const cv::Rect *roi = nullptr);

 private:
  ProcessManager preproc_;
  ProcessManager postproc_;
  wdr::dnn::BPUNets net_;
  const std::string model_name_;
  wdr::dnn::DequantScales dequant_scales_;
  std::vector<cv::Mat> out_feats_;
};

class YOLOv8 {
 public:
  YOLOv8(const std::string &appname, const wdr::json &cfg, int thread_num);

  std::vector<wdr::Box2D> run(const cv::Mat &img,
                              const std::vector<cv::Rect> &rois = {},
                              StatisticsTimeManager *stats = nullptr);

 private:
  const int thread_num_{1};
  std::vector<std::unique_ptr<BaseYOLOv8>> yolos_;

  std::vector<wdr::Box2D> run(int thread_index, const cv::Mat &img,
                              StatisticsTimeManager *stats = nullptr);

  tbb::concurrent_queue<cv::Rect> task_queue_;
};

}  // namespace wdr::apps
