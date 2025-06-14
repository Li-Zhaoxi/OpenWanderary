#pragma once

#include <opencv2/opencv.hpp>

namespace wdr::proc {

struct ImageAffineParms {
  double x_scale = 1;
  double y_scale = 1;
  int x_shift = 0;
  int y_shift = 0;
};

// 缩放图像
ImageAffineParms ResizeImage(const cv::Mat &img, const cv::Size &input_wh,
                             cv::Mat *out);

// LetterBox模式缩放图像
ImageAffineParms LetterBoxImage(const cv::Mat &img, const cv::Size &input_wh,
                                cv::Mat *out);

}  // namespace wdr::proc
