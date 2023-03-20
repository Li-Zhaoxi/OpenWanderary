#ifndef WDR_POSTPROC_H_
#define WDR_POSTPROC_H_
#include <Core/core.h>
namespace wdr
{
// 适用于二分类分割结果，后续有多分类需求，考虑优化这个函数或者改进这个函数
// 1x2xhxw->hxw
void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds);

}


#endif