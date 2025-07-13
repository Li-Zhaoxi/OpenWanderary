#pragma once
#include <map>
#include <string>
#include <vector>

#include <wanderary/utils/json_utils.h>

#include <opencv2/opencv.hpp>

namespace wdr::proc {

struct ImageAffineParms {
  ImageAffineParms() = default;
  ImageAffineParms(double x_scale, double y_scale, int x_shift, int y_shift)
      : x_scale(x_scale),
        y_scale(y_scale),
        x_shift(x_shift),
        y_shift(y_shift) {}
  double x_scale = 1;
  double y_scale = 1;
  int x_shift = 0;
  int y_shift = 0;

  wdr::json dump() const;
};

// 缩放图像
ImageAffineParms ResizeImage(const cv::Mat &img, const cv::Size &input_wh,
                             cv::Mat *out);

// LetterBox模式缩放图像
ImageAffineParms LetterBoxImage(const cv::Mat &img, const cv::Size &input_wh,
                                cv::Mat *out);

bool CheckFeatureDimValid(const cv::Mat &feat, const std::vector<int> &dims,
                          int dtype);

std::string LogFeatureDim(const cv::Mat &feat);

}  // namespace wdr::proc
