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

/**
 * \brief Letter Box是YOLO(You Only Look
 * Once)目标检测算法中用于图像预处理的关键技术，它解决了不同尺寸输入图像与模型固定输入尺寸之间的匹配问题。
 * \param[in] img 输入图像
 * \param[in] input_wh 输入图像的宽高
 * \param[out] out 输出图像
 * \return ImageAffineParms 缩放参数, 用于后续还原实际Box
 * \note Letter
 * Box是一种图像调整技术，主要用于深度学习领域中的物体检测任务，确保不同尺寸的输入图像能被模型接受。其核心思想是：
 * \note -
 * ‌保持原始宽高比‌：当输入图像的纵横比与模型所需不一致时，Letterbox会在图像的顶部和底部添加填充(通常是黑色或与背景色相同)，以保持纵横比的一致性‌
 * \note -
 * 按比例缩放‌：首先计算图像高宽分别与目标尺寸的缩放比例，取其中较小的一个以保持原图的宽高比‌
 * \note -‌
 * 填充处理‌：根据缩放比例计算调整后图像的无填充尺寸，然后计算需要在图像的宽和高方向上分别填充多少像素，以达到目标尺寸‌
 */
ImageAffineParms LetterBoxImage(const cv::Mat &img, const cv::Size &input_wh,
                                cv::Mat *out);

bool CheckFeatureDimValid(const cv::Mat &feat, const std::vector<int> &dims,
                          int dtype);

std::string LogFeatureDim(const cv::Mat &feat);

}  // namespace wdr::proc
