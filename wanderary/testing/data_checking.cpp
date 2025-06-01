#include "wanderary/testing/data_checking.h"

#include <string>
#include <vector>

#include <glog/logging.h>

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

}  // namespace wdr::testing
