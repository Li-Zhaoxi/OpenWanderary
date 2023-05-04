#ifndef WDR_POSTPROC_H_
#define WDR_POSTPROC_H_
#include <Core/core.h>
namespace wdr
{
// 设计思想，非必要，不设计新Class


// 适用于二分类分割结果，后续有多分类需求，考虑优化这个函数或者改进这个函数
// 1x2xhxw->hxw / 1xhxwx2->hxw
void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds);
// void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds, int idxc = -1);

// 目标检测 YoloV5，解析为 cv::Rect
// 绘制一批Rect

// 骨架提取
// 绘制骨架


// 

}


#endif