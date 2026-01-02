#pragma once
#include <string>
#include <vector>

#include <foxglove/CompressedImage.pb.h>
#include <foxglove/ImageAnnotations.pb.h>
#include <foxglove/PointsAnnotation.pb.h>

#include <opencv2/opencv.hpp>

#include "wanderary/structs/frame.h"

namespace wdr::msg {

void ConvertImageToMsg(const std::string &image_path, int64_t timestamp,
                       const std::string &frame_id,
                       foxglove::CompressedImage *msg);

void ConvertImageToMsg(const ImageData &image_data, int64_t timestamp,
                       const std::string &frame_id,
                       foxglove::CompressedImage *msg);

int64_t ConvertImageFromMsg(const foxglove::CompressedImage &msg, bool decode,
                            cv::Mat *image, std::string *format);

void ConvertPoint2dToMsg(const cv::Point2d &pt2d, foxglove::Point2 *msg);
void ConvertColorToMsg(const cv::Scalar &color, foxglove::Color *msg);

std::string ConvertBox2DToMsg(int64_t timestamp, const Box2D &box,
                              foxglove::PointsAnnotation *msg);

void ConvertBox2DLabelToTextMsg(int64_t timestamp, const Box2D &box,
                                const std::string &text,
                                foxglove::TextAnnotation *msg);

void ConvertBoxes2DToMsg(int64_t timestamp, const std::vector<Box2D> &boxes,
                         foxglove::ImageAnnotations *msg);

}  // namespace wdr::msg
