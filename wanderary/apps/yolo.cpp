#include "wanderary/apps/yolo.h"

#include <string>
#include <vector>

#include <glog/logging.h>

namespace wdr::apps {
using ProcessRecorder = wdr::proc::ProcessRecorder;

BaseYOLOv8::BaseYOLOv8(const std::string &appname, const wdr::json &cfg)
    : preproc_(appname + "/preprocess",
               wdr::GetData<wdr::json>(cfg, "preprocess")),
      postproc_(appname + "/postprocess",
                wdr::GetData<wdr::json>(cfg, "postprocess")),
      net_(wdr::GetData<std::string>(cfg, "model_path")),
      model_name_(wdr::GetData<std::string>(cfg, "model_name")) {
  dequant_scales_ = net_.GetDequantScales(model_name_);
  out_feats_.reserve(3);
}

std::vector<wdr::Box2D> BaseYOLOv8::run(cv::Mat *image,
                                        TimerManager *time_manager,
                                        const cv::Rect *roi) {
  CHECK(image->isContinuous());
  ProcessRecorder recorder;
  std::vector<wdr::Box2D> box2ds;

  {  // 预处理
    AutoScopeTimer timer("pre-process", time_manager);
    preproc_.Forward(image, &recorder);
  }

  {  // 推理
    AutoScopeTimer timer("bpu-inference", time_manager);
    recorder.dequant_scales = dequant_scales_;
    const std::vector<cv::Mat> input_mats = {
        cv::Mat(1, image->total(), CV_8UC1, image->data)};
    net_.Forward(model_name_, input_mats, &out_feats_);
  }

  {  // 后处理
    AutoScopeTimer timer("post-process", time_manager);
    postproc_.Forward(&out_feats_, &box2ds, &recorder);
  }

  if (roi) {
    for (auto &box2d : box2ds) {
      box2d.x_min += roi->x;
      box2d.y_min += roi->y;
    }
  }

  return box2ds;
}

YOLOv8::YOLOv8(const std::string &appname, const wdr::json &cfg, int thread_num)
    : thread_num_(thread_num) {
  CHECK_GT(thread_num, 0);
  for (int i = 0; i < thread_num; i++) {
    const std::string thread_name =
        "thread_" + std::to_string(i) + "/" + appname;
    yolos_.push_back(std::make_unique<BaseYOLOv8>(thread_name, cfg));
  }
}

std::vector<wdr::Box2D> YOLOv8::run(const cv::Mat &img,
                                    const std::vector<cv::Rect> &rois,
                                    StatisticsTimeManager *stats) {
  // 构造待处理的ROIs
  task_queue_.clear();
  for (const auto &roi : rois) task_queue_.push(roi);
  if (task_queue_.empty()) task_queue_.push(cv::Rect(0, 0, img.cols, img.rows));

  // 并行处理
  std::mutex mtx_;
  std::vector<wdr::Box2D> box2ds;
  tbb::parallel_for(  //
      tbb::blocked_range<size_t>(0, thread_num_),
      [&img, &mtx_, self = this, &stats,
       &box2ds](const tbb::blocked_range<size_t> &r) {
        int thread_index = r.begin();  // 获取当前线程的索引
        const auto res = self->run(thread_index, img, stats);
        std::lock_guard<std::mutex> lock(mtx_);
        box2ds.insert(box2ds.end(), res.begin(), res.end());
      });

  return box2ds;
}

std::vector<wdr::Box2D> YOLOv8::run(int thread_index, const cv::Mat &img,
                                    StatisticsTimeManager *stats) {
  auto &yolo_ = yolos_[thread_index];
  std::vector<wdr::Box2D> final_box2ds;
  while (!task_queue_.empty()) {
    cv::Rect roi;
    if (!task_queue_.try_pop(roi)) continue;
    cv::Mat roi_img;
    img(roi).copyTo(roi_img);
    TimerManager time_manager;
    const auto res = yolo_->run(&roi_img, &time_manager, &roi);
    final_box2ds.insert(final_box2ds.end(), res.begin(), res.end());
    if (stats) stats->add(time_manager);
  }
  return final_box2ds;
}

}  // namespace wdr::apps
