#ifndef WDR_POSTPROC_H_
#define WDR_POSTPROC_H_
#include <Core/core.h>

// 后处理函数集
// 设计思想：非不要不设计新Class
// 函数规则：每个后处理函数以analyze开头，而多Batch的后处理需求以analyzeBatch开头
//          有可视化需求的会以draw开头

// 目前已提供的后处理函数：
// 二分类网络
// yolo类的后处理

namespace wdr
{
  // 设计思想，非必要，不设计新Class

  // 适用于二分类分割结果，后续有多分类需求，考虑优化这个函数或者改进这个函数
  // 1x2xhxw->hxw / 1xhxwx2->hxw

  // 二分类优化，判断是否为3维，如果是3d的话默认b为1，检查是否为HWC模式。
  void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds);
  // void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds, int idxc = -1);

  // Yolo后处理
  // src支持的格式: （rows, cols, channels() == 1）
  // 单batch模式： (H, W), (H, W, 1), (1, H, W, 1)
  // batch模式： (b, H, W), (b, H, W, 1), (H, W), (H, W, 1)
  // void analyzeYoloResult(const cv::Mat &src);
  // void analyzeBatchYoloResult(const cv::Mat &src);
  // 目标检测 YoloV5，解析为 cv::Rect
  // 绘制一批Rect

  // 骨架提取
  // 绘制骨架

  //

}

#endif