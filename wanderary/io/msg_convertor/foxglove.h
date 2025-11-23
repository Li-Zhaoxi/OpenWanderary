#pragma once
#include <string>

#include <foxglove/CompressedImage.pb.h>

#include <opencv2/opencv.hpp>

namespace wdr::msg {

void ConvertImageToMsg(const std::string &image_path, int64_t timestamp,
                       const std::string &frame_id,
                       foxglove::CompressedImage *msg);

int64_t ConvertImageFromMsg(const foxglove::CompressedImage &msg, bool decode,
                            cv::Mat *image, std::string *format);

}  // namespace wdr::msg
