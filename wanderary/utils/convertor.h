#pragma once

#include <opencv2/opencv.hpp>

namespace wdr {

void BGRToNV12(const cv::Mat &bgr, cv::Mat *nv12);

void NV12ToYUV444(const cv::Mat &nv12, const cv::Size size, cv::Mat *yuv444);

}  // namespace wdr
