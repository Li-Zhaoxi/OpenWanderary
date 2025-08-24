#pragma once
#include <vector>

#include <wanderary/structs/box.h>

#include <opencv2/opencv.hpp>

namespace wdr::vis {

void DrawBoxes2D(const std::vector<wdr::Box2D> &boxes, cv::Mat *img);

}  // namespace wdr::vis
