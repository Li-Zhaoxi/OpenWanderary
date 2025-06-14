#pragma once
#include <map>
#include <vector>

#include <wanderary/process/process_base.h>
#include <wanderary/utils/enum_traits.h>
#include <wanderary/utils/json_utils.h>

namespace wdr::proc {

struct CvtYoloConfig {
  int class_num_ = 80;
  int reg_num_ = 16;
  double nms_thres_ = 0.7;
  double score_thres_ = 0.25;

  // 以下变量不需要指定
  double conf_thres_ = 0;
};

class ConvertYoloFeature : public ProcessBase {
 public:
  explicit ConvertYoloFeature(const utils::json &cfg);

  void Forward(const std::vector<cv::Mat> &feats, std::vector<cv::Rect> *box2ds,
               ProcessRecorder *recorder = nullptr) const override;

 private:
  float ConvertOne(const float *fscore, const int *fbox, const float *boxscles,
                   cv::Rect *box2d) const;
  CvtYoloConfig cfg_;
  std::vector<float> box_weights_;
};

}  // namespace wdr::proc
