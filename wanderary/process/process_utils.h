#pragma once
#include <map>
#include <string>
#include <vector>

#include <wanderary/utils/json_utils.h>

#include <opencv2/opencv.hpp>

namespace wdr::proc {

struct ImageAffineParms {
  double x_scale = 1;
  double y_scale = 1;
  int x_shift = 0;
  int y_shift = 0;

  wdr::utils::json dump() const;
};

struct DequantScales {
  std::map<int, std::vector<float>> de_scales;
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
