#ifndef WDR_CORE_H_
#define WDR_CORE_H_

#include <opencv2/opencv.hpp>

namespace wdr
{

  ////////////// basic
  void RootRequired();

  void get_rgb_image(const std::string &imgpath, cv::Mat &img);
  void get_bgr_image(const std::string &imgpath, cv::Mat &img);
  std::vector<size_t> get_shape(cv::Mat &mat);

  void imequalresize(const cv::Mat &img, const cv::Size &target_size, const cv::Scalar &pad_value, cv::Mat &pad_image);

  // class

  // refer to https://answers.opencv.org/question/226929/how-could-i-change-memory-layout-from-hwc-to-chw/
  void hwc_to_chw(cv::InputArray src, cv::OutputArray dst);
  void chw_to_hwc(cv::InputArray src, cv::OutputArray dst);
  void makeContinuous(const cv::Mat &src, cv::Mat &dst);

}

#endif