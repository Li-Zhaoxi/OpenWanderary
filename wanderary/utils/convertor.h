#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

namespace wdr {

void BGRToNV12(const cv::Mat &bgr, cv::Mat *nv12);

void NV12ToYUV444(const cv::Mat &nv12, const cv::Size size, cv::Mat *yuv444);

/**
 * \brief 基于图像窗口的模式从图像中裁剪出多个ROI
 * \param img_size 图像大小
 * \param crop_size 裁剪大小
 * \param offset 偏移量
 * \param drop_gap 如果末尾的roi不完整，是否丢弃
 */
std::vector<cv::Rect> ImageCropROIs(const cv::Size &img_size,
                                    const cv::Size &crop_size,
                                    const cv::Size &offset, bool drop_gap);

}  // namespace wdr
