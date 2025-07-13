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
  float nms_thres_ = 0.7;
  float score_thres_ = 0.25;
  std::map<int, float> box_scales_;

  // 以下变量不需要指定
  double conf_thres_ = 0;
};

class ConvertYoloFeature : public ProcessBase {
 public:
  explicit ConvertYoloFeature(const json &cfg);
  ~ConvertYoloFeature() override;

  void Forward2D(const std::vector<cv::Mat> &feats,
                 std::vector<wdr::Box2D> *box2ds,
                 ProcessRecorder *recorder = nullptr) const;
  void Forward(std::vector<cv::Mat> *feats, std::vector<wdr::Box2D> *box2ds,
               ProcessRecorder *recorder) const override;

 private:
  float ConvertOne(int idxh, int idxw, const float *fscore, const int *fbox,
                   const float *boxdescles, float boxscale, cv::Rect2d *box2d,
                   int *idx_max_score) const;

  CvtYoloConfig cfg_;
  std::vector<float> box_weights_;
};

}  // namespace wdr::proc
