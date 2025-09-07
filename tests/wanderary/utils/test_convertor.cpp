#include <string>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/testing/data_checking.h"
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

cv::Mat DrawROISs(const cv::Mat &img, const std::vector<cv::Rect> &rois) {
  cv::Mat res;
  img.copyTo(res);
  cv::RNG rng(0);
  for (const auto &roi : rois)
    cv::rectangle(res, roi,
                  cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                             rng.uniform(0, 255)),
                  2);
  return res;
}

TEST(Convertor, ImageCropROIs) {
  const std::string imgpath = "../../test_data/utils/street.jpg";
  const cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);
  {
    const auto rois = wdr::ImageCropROIs(img.size(), cv::Size(640, 640),
                                         cv::Size(640, 640), false);
    const std::vector<cv::Rect> roisgt = {
        cv::Rect(0, 0, 640, 640),     cv::Rect(640, 0, 640, 640),
        cv::Rect(1280, 0, 640, 640),  cv::Rect(0, 640, 640, 640),
        cv::Rect(640, 640, 640, 640), cv::Rect(1280, 640, 640, 640)};
    wdr::testing::Check(rois, roisgt);
    cv::imwrite("ImageCropROIs_nooffset.png", DrawROISs(img, rois));
  }
  {
    const auto rois = wdr::ImageCropROIs(img.size(), cv::Size(640, 640),
                                         cv::Size(600, 600), false);
    const std::vector<cv::Rect> roisgt = {
        cv::Rect(0, 0, 640, 640),      cv::Rect(600, 0, 640, 640),
        cv::Rect(1200, 0, 640, 640),   cv::Rect(1800, 0, 120, 640),
        cv::Rect(0, 600, 640, 640),    cv::Rect(600, 600, 640, 640),
        cv::Rect(1200, 600, 640, 640), cv::Rect(1800, 600, 120, 640),
        cv::Rect(0, 1200, 640, 80),    cv::Rect(600, 1200, 640, 80),
        cv::Rect(1200, 1200, 640, 80), cv::Rect(1800, 1200, 120, 80)};
    cv::imwrite("ImageCropROIs_offset_nodrop.png", DrawROISs(img, rois));
    wdr::testing::Check(rois, roisgt);
  }
  {
    const auto rois = wdr::ImageCropROIs(img.size(), cv::Size(640, 640),
                                         cv::Size(600, 600), true);
    const std::vector<cv::Rect> roisgt = {
        cv::Rect(0, 0, 640, 640),     cv::Rect(600, 0, 640, 640),
        cv::Rect(1200, 0, 640, 640),  cv::Rect(0, 600, 640, 640),
        cv::Rect(600, 600, 640, 640), cv::Rect(1200, 600, 640, 640)};
    cv::imwrite("ImageCropROIs_offset_drop.png", DrawROISs(img, rois));
    wdr::testing::Check(rois, roisgt);
  }
}
