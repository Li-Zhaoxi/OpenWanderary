#pragma once
#include <string>
#include <vector>

#include <wanderary/structs/box.h>

#include <opencv2/opencv.hpp>

namespace wdr::vis {

void DrawBoxes2D(const std::vector<wdr::Box2D> &boxes, cv::Mat *img);

class Box2DDrawer {
 public:
  explicit Box2DDrawer(int class_num,
                       const std::vector<std::string> &class_names = {});
  void draw(const std::vector<wdr::Box2D> &boxes, cv::Mat *img);

 private:
  int class_num_;
  std::vector<std::string> class_names_;
  std::vector<cv::Scalar> colors_;
};

}  // namespace wdr::vis
