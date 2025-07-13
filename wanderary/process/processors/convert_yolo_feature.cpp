#include "wanderary/process/processors/convert_yolo_feature.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include "wanderary/utils/common_utils.h"
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

ConvertYoloFeature::ConvertYoloFeature(const json &cfg)
    : ProcessBase("ConvertYoloFeature") {
  cfg_.class_num_ = wdr::GetData<int>(cfg, "class_num");
  cfg_.reg_num_ = wdr::GetData<int>(cfg, "reg_num");
  cfg_.nms_thres_ = wdr::GetData<float>(cfg, "nms_thres");
  cfg_.score_thres_ = wdr::GetData<float>(cfg, "score_thres");

  const auto scale_data =
      wdr::GetData<std::map<std::string, float>>(cfg, "box_scales");
  for (auto &it : scale_data) cfg_.box_scales_[std::stoi(it.first)] = it.second;
  CHECK_EQ(cfg_.box_scales_.size(), 3);

  cfg_.conf_thres_ = -std::log(1.0 / cfg_.score_thres_ - 1.0);
  box_weights_ = BoxWeights(cfg_.reg_num_);
}

ConvertYoloFeature::~ConvertYoloFeature() {}

void ConvertYoloFeature::Forward2D(const std::vector<cv::Mat> &feats,
                                   std::vector<wdr::Box2D> *box2ds,
                                   ProcessRecorder *recorder) const {
  box2ds->clear();
  // 有效性校验. 转换Box时需要用到recorder的内容
  const int feat_num = 6;
  DCHECK_EQ(feats.size(), feat_num);
  DCHECK(recorder != nullptr && recorder->affine.has_value() &&
         recorder->dequant_scales.has_value());
  const auto &descales = recorder->dequant_scales.value();
  const auto &boxscales = cfg_.box_scales_;
  for (int i = 0; i < feat_num; i += 2) {
    const auto &feat = feats[i];
    DCHECK(CheckFeatureDimValid(feat, {1, -1, -1, cfg_.class_num_},
                                cv::DataType<float>::type))
        << "Invalid feature. Idx: " << i
        << ", feat info: " << LogFeatureDim(feat);
  }
  for (int i = 1; i < feat_num; i += 2) {
    const auto &feat = feats[i];
    DCHECK(CheckFeatureDimValid(feat, {1, -1, -1, cfg_.reg_num_ * 4},
                                cv::DataType<int>::type))
        << "Invalid feature. Idx: " << i
        << ", feat info: " << LogFeatureDim(feat);
    DCHECK(wdr::contains(descales.de_scales, i))
        << "Cannot find dequant scale for feature idx: " << i;
    DCHECK(wdr::contains(cfg_.box_scales_, i))
        << "Cannot find box scale for feature idx: " << i;
  }

  // TBB多线程处理
  std::mutex mtx;
  std::vector<std::vector<float>> scores(cfg_.class_num_);
  std::vector<std::vector<cv::Rect2d>> boxes(cfg_.class_num_);
  auto fun_proc_feat = [&mtx, &feats, &descales, &boxscales, this, &scores,
                        &boxes](int idx) {
    const int idx_fscore = idx * 2;
    const int idx_fbox = idx * 2 + 1;
    const auto &fscore = feats[idx_fscore];
    const int frows = fscore.size[1];
    const int fcols = fscore.size[2];
    const int total = frows * fcols;

    const float *pfscore = fscore.ptr<float>(0);
    const int *pfbox = feats[idx_fbox].ptr<int>(0);
    const float *pdescales = descales.de_scales.at(idx_fbox).data();
    const float boxscale = boxscales.at(idx_fbox);

    auto fun_proc_one = [&mtx, fcols, pfscore, pfbox, pdescales, boxscale, this,
                         &scores, &boxes](int idx) {
      const int idxh = idx / fcols;
      const int idxw = idx % fcols;
      const float *cur_pfscore = pfscore + idx * this->cfg_.class_num_;
      const int *cur_pfbox = pfbox + idx * this->cfg_.reg_num_ * 4;
      cv::Rect2d box2d;
      int idx_max_score = 0;
      const float score =
          this->ConvertOne(idxh, idxw, cur_pfscore, cur_pfbox, pdescales,
                           boxscale, &box2d, &idx_max_score);
      if (score < 0) return;
      mtx.lock();
      scores[idx_max_score].push_back(score);
      boxes[idx_max_score].push_back(box2d);
      mtx.unlock();
    };
    tbb::parallel_for(0, total, fun_proc_one);
  };
  tbb::parallel_for(0, static_cast<int>(feats.size() / 2), fun_proc_feat);

  // NMS
  auto fun_nms = [box2ds, affine = &recorder->affine.value(), &scores, &boxes,
                  &mtx, this](int idx) {
    std::vector<int> indices;
    const auto &usage_boxes = boxes[idx];
    const auto &usage_scores = scores[idx];
    if (usage_boxes.empty()) return;
    cv::dnn::NMSBoxes(usage_boxes, usage_scores, this->cfg_.score_thres_,
                      this->cfg_.nms_thres_, indices);

    mtx.lock();
    for (auto &idx_nms : indices) {
      wdr::Box2D tmp;
      const auto &box2d = usage_boxes[idx_nms];
      tmp.score = usage_scores[idx_nms];
      tmp.x_min = (box2d.x - affine->x_shift) / affine->x_scale;
      tmp.y_min = (box2d.y - affine->y_shift) / affine->y_scale;
      tmp.w = box2d.width / affine->x_scale;
      tmp.h = box2d.height / affine->y_scale;
      tmp.label_id = idx;

      box2ds->push_back(std::move(tmp));
    }
    mtx.unlock();
  };
  tbb::parallel_for(0, static_cast<int>(scores.size()), fun_nms);
}

float ConvertYoloFeature::ConvertOne(int idxh, int idxw, const float *fscore,
                                     const int *fbox, const float *boxdescles,
                                     float boxscale, cv::Rect2d *box2d,
                                     int *idx_max_score) const {
  // 计算fscore最大值以及最大值索引
  int idx = -1;
  float max_score = cfg_.conf_thres_ - 1.0f;
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
  *idx_max_score = idx;

  // 反量化Box特征
  cv::Mat boxfp(1, cfg_.reg_num_ * 4, CV_32FC1);
  float *data = boxfp.ptr<float>(0);
  for (int i = 0; i < boxfp.cols; i++) {
    data[i] = static_cast<float>(fbox[i]) * boxdescles[i];
  }

  const float *w = box_weights_.data();
  cv::Vec4f ltrb_indices;
  for (int i = 0; i < 4; i++) {
    float *ptr = data + i * cfg_.reg_num_;
    wdr::softmax<float>(ptr, cfg_.reg_num_);
    ltrb_indices[i] = wdr::weight_sum(ptr, w, cfg_.reg_num_);
  }

  box2d->x = (idxw + 0.5 - ltrb_indices[0]) * boxscale;
  box2d->y = (idxh + 0.5 - ltrb_indices[1]) * boxscale;
  box2d->width = (ltrb_indices[2] + ltrb_indices[0]) * boxscale;
  box2d->height = (ltrb_indices[3] + ltrb_indices[1]) * boxscale;

  return max_score;
}

void ConvertYoloFeature::Forward(std::vector<cv::Mat> *feats,
                                 std::vector<wdr::Box2D> *box2ds,
                                 ProcessRecorder *recorder) const {
  this->Forward2D(*feats, box2ds, recorder);
}

REGISTER_DERIVED_CLASS(ProcessBase, ConvertYoloFeature)

}  // namespace wdr::proc
