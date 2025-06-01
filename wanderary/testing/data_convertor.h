#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace wdr::testing {

std::vector<uchar> Convertor(const cv::Mat &img);

}  // namespace wdr::testing
