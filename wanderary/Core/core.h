#ifndef WDR_CORE_H_
#define WDR_CORE_H_


#include <opencv2/opencv.hpp>


namespace wdr
{

  // class 

// refer to https://answers.opencv.org/question/226929/how-could-i-change-memory-layout-from-hwc-to-chw/
void hwc_to_chw(cv::InputArray src, cv::OutputArray dst);
void chw_to_hwc(cv::InputArray src, cv::OutputArray dst);
void makeContinuous(const cv::Mat &src, cv::Mat &dst);

}


#endif