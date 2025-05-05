#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/utils/convertor.h"

TEST(Convertor, BGR2NV12) {
  const std::string imgpath = "../../test_data/media/zidane.jpg";
  const std::string gtpath = "../../test_data/utils/zidane_nv12.png";

  const cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);
  const cv::Mat imggt = cv::imread(gtpath, cv::IMREAD_COLOR);

  cv::Mat nv12;
  wdr::BGRToNV12(img, &nv12);

  cv::Mat yuv444;
  wdr::NV12ToYUV444(nv12, img.size(), &yuv444);

  cv::Mat res;
  cv::cvtColor(yuv444, res, cv::COLOR_YUV2BGR);

  const auto gtdiff = cv::sum(cv::abs(res - imggt));

  EXPECT_NEAR(gtdiff[0], 0, 1e-6);
  EXPECT_NEAR(gtdiff[1], 0, 1e-6);
  EXPECT_NEAR(gtdiff[2], 0, 1e-6);
}
