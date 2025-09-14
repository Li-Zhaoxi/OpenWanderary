#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

namespace wdr::vis {

/**
 * \brief 根据背景色计算最佳文字颜色(黑或白)
 * \param[in] bg_color 背景色
 * \return 文字颜色
 * \note 会根据背景色计算相对亮度 (ITU-R BT.709标准)
 */
cv::Scalar GetTextColor(const cv::Scalar& bg_color);

/**
 * \brief 获取每个类别的颜色
 * \param[in] num_classes 类别数量
 * \return 每个类别的颜色
 */
std::vector<cv::Scalar> GetColorMap(int num_classes);

}  // namespace wdr::vis
