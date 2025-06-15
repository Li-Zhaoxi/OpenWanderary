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

void Check(const cv::Mat &pred, const cv::Mat &gt, double eps,
           const std::string &msg) {
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

}  // namespace wdr::testing
