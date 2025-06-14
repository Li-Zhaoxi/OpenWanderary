#include "wanderary/process/processors/convert_yolo_feature.h"

#include <vector>

#include "wanderary/utils/math_utils.h"

namespace wdr::proc {

namespace {

std::vector<float> BoxWeights(int num) {
  std::vector<float> weights(num);
  for (int i = 0; i < num; ++i) {
    weights[i] = i;
  }
  return weights;
}

}  // namespace

ConvertYoloFeature::ConvertYoloFeature(const utils::json &cfg)
    : ProcessBase("ConvertYoloFeature") {
  cfg_.class_num_ = wdr::utils::GetData<int>(cfg, "class_num");
  cfg_.reg_num_ = wdr::utils::GetData<int>(cfg, "reg_num");
  cfg_.nms_thres_ = wdr::utils::GetData<float>(cfg, "nms_thres");
  cfg_.score_thres_ = wdr::utils::GetData<float>(cfg, "score_thres");

  cfg_.conf_thres_ = -std::log(1.0 / cfg_.score_thres_ - 1.0);
  box_weights_ = BoxWeights(cfg_.reg_num_);
}

void ConvertYoloFeature::Forward(const std::vector<cv::Mat> &feats,
                                 std::vector<cv::Rect> *box2ds,
                                 ProcessRecorder *recorder) const {
  // 有效性校验. 转换Box时需要用到recorder的内容
  const int feat_num = 6;
  DCHECK_EQ(feats.size(), feat_num);
  DCHECK(recorder != nullptr && recorder->affine.has_value());
  for (int i = 0; i < feat_num; i += 2) {
    const auto &feat = feats[i];
    CHECK(CheckFeatureDimValid(feat, {1, -1, -1, cfg_.class_num_},
                               cv::DataType<float>::type))
        << "Invalid feature. Idx: " << i
        << ", feat info: " << LogFeatureDim(feat);
  }
  for (int i = 1; i < feat_num; i += 2) {
    const auto &feat = feats[i];
    CHECK(CheckFeatureDimValid(feat, {1, -1, -1, cfg_.reg_num_ * 4},
                               cv::DataType<int32_t>::type))
        << "Invalid feature. Idx: " << i
        << ", feat info: " << LogFeatureDim(feat);
  }

  // 2. 补充单个数据的转换方法
  // 3. tbb多线程优化
}

float ConvertYoloFeature::ConvertOne(const float *fscore, const int *fbox,
                                     const float *boxscles,
                                     cv::Rect *box2d) const {
  // 计算fscore最大值以及最大值索引
  int idx = -1;
  float max_score = -1.0f;
  for (int i = 0; i < cfg_.class_num_; i++) {
    const float val = fscore[i];
    if (val < cfg_.conf_thres_) continue;
    if (val > max_score) {
      max_score = val;
      idx = i;
    }
  }
  if (idx < 0) return -1.0f;
  max_score = 1.0f / (1.0f + std::exp(-max_score));

  // 反量化Box特征
  cv::Mat boxfp(1, cfg_.reg_num_ * 4, CV_32FC1);
  float *data = boxfp.ptr<float>(0);
  for (int i = 0; i < boxfp.cols; i++) {
    data[i] = static_cast<float>(fbox[i]) * boxscles[i];
  }

  const float *w = box_weights_.data();
  cv::Vec4f ltrb_indices;
  for (int i = 0; i < 4; i++) {
    float *ptr = data + i * cfg_.reg_num_;
    wdr::softmax<float>(ptr, cfg_.reg_num_);
    ltrb_indices[i] = wdr::weight_sum(ptr, w, cfg_.reg_num_);
  }

  return max_score;
}

REGISTER_DERIVED_CLASS(ProcessBase, ConvertYoloFeature)

}  // namespace wdr::proc
