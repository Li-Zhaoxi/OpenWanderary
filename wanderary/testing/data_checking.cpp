#include "wanderary/testing/data_checking.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include <Eigen/Dense>

#include "wanderary/utils/common_utils.h"

namespace wdr::testing {

namespace {

std::vector<Eigen::Vector<double, 5>> ConstructBox2DFeatures(
    const std::vector<wdr::Box2D> &box2ds) {
  Eigen::Vector2d tmp;
  std::vector<Eigen::Vector<double, 5>> features;

  for (const auto &box2d : box2ds) {
    Eigen::Vector<double, 5> feature;
    feature << box2d.x_min, box2d.y_min, box2d.w, box2d.h, box2d.score;
    features.push_back(std::move(feature));
  }

  return features;
}

}  // namespace

void CheckGeneralMat(const cv::Mat &pred, const cv::Mat &gt, double eps,
                     const std::string &msg) {
  EXPECT_EQ(pred.rows, gt.rows) << msg;
  EXPECT_EQ(pred.cols, gt.cols) << msg;
  EXPECT_EQ(pred.size(), gt.size()) << msg;
  EXPECT_EQ(pred.type(), gt.type()) << msg;
  EXPECT_EQ(pred.channels(), gt.channels()) << msg;

  if (pred.size() != gt.size() || pred.type() != gt.type() ||
      pred.channels() != gt.channels())
    return;

  cv::Mat diff;
  if (gt.depth() == CV_32F || gt.depth() == CV_64F) {
    cv::absdiff(pred, gt, diff);
    EXPECT_EQ(cv::countNonZero(diff > eps), 0) << msg;
  } else {
    cv::compare(pred, gt, diff, cv::CMP_NE);
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    for (int i = 0; i < channels.size(); i++)
      EXPECT_EQ(cv::countNonZero(channels[i]), 0)
          << "(chl idx: " << i << ")->" << msg;
  }
}

void CheckNonGeneralMat(const cv::Mat &pred, const cv::Mat &gt, double eps,
                        const std::string &msg) {
  CHECK_EQ(gt.rows, -1) << msg;
  CHECK_EQ(gt.cols, -1) << msg;
  // 维度校验
  EXPECT_EQ(pred.size.dims(), gt.size.dims()) << msg;
  if (pred.size.dims() != gt.size.dims()) return;
  for (int i = 0; i < pred.size.dims(); i++) {
    EXPECT_EQ(pred.size[i], gt.size[i]) << "index: " << i << " " << msg;
    if (pred.size[i] != gt.size[i]) return;
  }

  // 数值校验
  const int total = gt.total();
  CheckGeneralMat(cv::Mat(1, total, pred.type(), pred.data),
                  cv::Mat(1, total, gt.type(), gt.data), eps,
                  "CheckNonGeneralMat " + msg);
}

void Check(const std::set<std::string> &pred, const std::set<std::string> &gt,
           const std::string &msg) {
  EXPECT_EQ(pred.size(), gt.size()) << msg;
  if (pred.size() != gt.size()) return;
  auto tmp = pred;
  for (const auto &item : gt) {
    EXPECT_TRUE(wdr::contains(pred, item))
        << "Cannot find " << item << ". " << msg;
    tmp.erase(item);
  }

  if (!tmp.empty()) {
    std::stringstream ss;
    ss << "Pred Remaining: ";
    for (const auto &item : tmp) ss << item << ",";
    EXPECT_TRUE(false) << ss.str() << msg;
  }
}

void Check(const wdr::Box2D &pred, const wdr::Box2D &gt, double eps,
           const std::string &msg) {
  EXPECT_NEAR(pred.score, gt.score, eps) << msg;
  EXPECT_NEAR(pred.x_min, gt.x_min, eps) << msg;
  EXPECT_NEAR(pred.y_min, gt.y_min, eps) << msg;
  EXPECT_NEAR(pred.w, gt.w, eps) << msg;
  EXPECT_NEAR(pred.h, gt.h, eps) << msg;
  EXPECT_EQ(pred.label_id, gt.label_id) << msg;
}

void UnorderedCheck(const std::vector<wdr::Box2D> &pred,
                    const std::vector<wdr::Box2D> &gt, double eps,
                    const std::string &msg) {
  EXPECT_EQ(pred.size(), gt.size()) << msg;
  if (pred.size() != gt.size()) return;
  const auto pred_feats = ConstructBox2DFeatures(pred);
  const auto gt_feats = ConstructBox2DFeatures(gt);

  for (int idxgt = 0; idxgt < gt.size(); idxgt++) {
    const auto &gt_feat = gt_feats[idxgt];
    double min_dist = 1e10;
    int min_idx = -1;
    for (int idxpred = 0; idxpred < pred.size(); idxpred++) {
      const auto &pred_feat = pred_feats[idxpred];
      const double dist = (gt_feat - pred_feat).norm();
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = idxpred;
      }
    }
    Check(pred[min_idx], gt[idxgt], eps, msg);
  }
}

void Check(const wdr::json &pred, const wdr::json &gt, double eps,
           const std::string &msg) {
  // 数据类型校验
  EXPECT_EQ(pred.type_name(), gt.type_name()) << msg;
  if (pred.type_name() != gt.type_name()) return;
  const auto data_type = pred.type();

  if (data_type == wdr::json::value_t::null) {
    return;
  } else if (data_type == wdr::json::value_t::string) {
    EXPECT_EQ(pred.get<std::string>(), gt.get<std::string>()) << msg;
  } else if (data_type == wdr::json::value_t::boolean) {
    EXPECT_EQ(pred.get<bool>(), gt.get<bool>()) << msg;
  } else if (data_type == wdr::json::value_t::number_float) {
    EXPECT_NEAR(pred.get<double>(), gt.get<double>(), eps) << msg;
  } else if (data_type == wdr::json::value_t::number_integer) {
    EXPECT_EQ(pred.get<int>(), gt.get<int>()) << msg;
  } else if (data_type == wdr::json::value_t::number_unsigned) {
    EXPECT_EQ(pred.get<unsigned int>(), gt.get<unsigned int>()) << msg;
  } else if (data_type == wdr::json::value_t::object) {
    EXPECT_EQ(pred.size(), gt.size()) << msg;
    if (pred.size() != gt.size()) return;

    for (const auto &item : pred.items()) {
      EXPECT_TRUE(gt.contains(item.key())) << msg;
      if (!gt.contains(item.key())) continue;
      std::stringstream ss;
      ss << "key: " << item.key() << "(" << msg << ")";
      Check(item.value(), gt[item.key()], eps, ss.str());
    }
  } else if (data_type == wdr::json::value_t::array) {
    EXPECT_EQ(pred.size(), gt.size()) << msg;
    if (pred.size() != gt.size()) return;
    const int num = pred.size();

    for (int i = 0; i < num; ++i) {
      std::stringstream ss;
      ss << "index " << i << "(" << msg << ")";
      Check(pred[i], gt[i], eps, ss.str());
    }
  } else {
    LOG(INFO) << "Not implemented for type " << pred.type_name() << ". " << msg;
  }
}

}  // namespace wdr::testing
