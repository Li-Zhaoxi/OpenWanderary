#include "wanderary/testing/data_checking.h"

#include <set>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "wanderary/utils/common_utils.h"

namespace wdr::testing {

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

}  // namespace wdr::testing
