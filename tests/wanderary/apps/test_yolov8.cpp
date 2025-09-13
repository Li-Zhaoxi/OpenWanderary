#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "wanderary/apps/yolo.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/utils/convertor.h"
#include "wanderary/utils/json_utils.h"

using YOLOv8 = wdr::apps::YOLOv8;

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

TEST(YOLOv8, LargeImage) {
  const std::string imgpath = "../../test_data/utils/street.jpg";
  const std::string cfgpath = "../../test_data/apps/yolov8_config.json";
  const std::string gtpath = "yolov8_result.json";

  // 构造GT
  const auto gtdata = wdr::LoadJson(gtpath);
  std::vector<wdr::Box2D> box2ds_gt;
  for (const auto &item : gtdata) box2ds_gt.push_back(wdr::Box2D::load(item));

  // 初始化
  YOLOv8 yolo("yolov8", wdr::LoadJson(cfgpath), 1);
  const cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);

  for (int i = 0; i < 10; i++) {
    const auto rois = wdr::ImageCropROIs(img.size(), cv::Size(640, 640),
                                         cv::Size(640, 640), false);
    const auto bbox2ds = yolo.run(img, rois);

    wdr::testing::UnorderedCheck(bbox2ds, box2ds_gt, 1e-4);
  }
}
