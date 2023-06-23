#ifndef WDR_PREPROC_H_
#define WDR_PREPROC_H_
#include <Core/core.h>

namespace wdr
{
  void preprocess_onboard_NHWC(const cv::Mat img, int modelh, int modelw, cv::Mat &datain);
  void preprocess_onboard_YoloV5BGRNHWC(const cv::Mat img, int modelh, int modelw, cv::Mat &datain);
}

#endif