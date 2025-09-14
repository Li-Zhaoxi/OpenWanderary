#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "wanderary/apps/yolo.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/utils/convertor.h"
#include "wanderary/utils/json_utils.h"
#include "wanderary/visualization/draw_boxes.h"

using YOLOv8 = wdr::apps::YOLOv8;
using StatisticsTimeManager = wdr::StatisticsTimeManager;
using TimerManager = wdr::TimerManager;
using Box2DDrawer = wdr::vis::Box2DDrawer;

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
  const std::string gtpath = "../../test_data/utils/street_box2d_gt.json";
  const std::string names_path = "../../test_data/tiny_coco/type_names.json";

  // 构造GT
  const auto gtdata = wdr::LoadJson(gtpath);
  std::vector<wdr::Box2D> box2ds_gt;
  for (const auto &item : gtdata) box2ds_gt.push_back(wdr::Box2D::load(item));

  // 初始化
  std::vector<std::string> class_names(80);
  const auto names = wdr::LoadJson(names_path);
  for (const auto &item : names.items()) {
    class_names[std::stoi(item.key())] = item.value();
  }
  Box2DDrawer drawer(80, class_names);
  YOLOv8 yolo("yolov8", wdr::LoadJson(cfgpath), 1);
  const cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);

  StatisticsTimeManager stats_manager;
  for (int i = 0; i < 10; i++) {
    const auto rois = wdr::ImageCropROIs(img.size(), cv::Size(640, 640),
                                         cv::Size(640, 640), false);
    TimerManager time_manager;
    time_manager.start("full-pipeline");
    const auto bbox2ds = yolo.run(img, rois, &stats_manager);
    time_manager.stop("full-pipeline");
    stats_manager.add(time_manager);
    wdr::testing::UnorderedCheck(bbox2ds, box2ds_gt, 1e-4);

    if (i == 0) {
      cv::Mat vis;
      img.copyTo(vis);
      drawer.draw(bbox2ds, &vis);
      cv::imwrite("vis_yolov8_large_image.jpg", vis);
    }
  }
  stats_manager.printStatistics();
}
