// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/data_transformer.h"

#include <iostream>

#include "glog/logging.h"

void mobilenetv1_transformers(ImageTensor *image_tensor,
                              cv::Mat &output_mat,
                              cv::Mat &input_mat) {
  cv::Mat resize_mat;
  short_side_resize(resize_mat, input_mat, 256);
  centor_crop(output_mat, resize_mat, 224);
}

void mobilenetv2_transformers(ImageTensor *image_tensor,
                              cv::Mat &output_mat,
                              cv::Mat &input_mat) {
  cv::Mat resize_mat;
  short_side_resize(resize_mat, input_mat, 256);
  centor_crop(output_mat, resize_mat, 224);
}

void resnet18_transformers(ImageTensor *image_tensor,
                           cv::Mat &output_mat,
                           cv::Mat &input_mat) {
  cv::Mat resize_mat;
  short_side_resize(resize_mat, input_mat, 256);
  centor_crop(output_mat, resize_mat, 224);
}

void resnet50_transformers(ImageTensor *image_tensor,
                           cv::Mat &output_mat,
                           cv::Mat &input_mat) {
  cv::Mat resize_mat;
  short_side_resize(resize_mat, input_mat, 256);
  centor_crop(output_mat, resize_mat, 224);
}

void googlenet_transformers(ImageTensor *image_tensor,
                            cv::Mat &output_mat,
                            cv::Mat &input_mat) {
  cv::Mat resize_mat;
  short_side_resize(resize_mat, input_mat, 256);
  centor_crop(output_mat, resize_mat, 224);
}

void efficientnet_transformers(ImageTensor *image_tensor,
                               cv::Mat &output_mat,
                               cv::Mat &input_mat) {
  cv::Mat crop_mat;
  padded_center_crop(crop_mat, input_mat, 224.0, 32);
  cv::resize(crop_mat, output_mat, output_mat.size(), 0, 0);
}

void yolov2_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat) {
  padding_resize(image_tensor, output_mat, input_mat);
}

void yolov3_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat) {
  padding_resize(image_tensor, output_mat, input_mat);
}

void yolov5_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat) {
  padding_resize(image_tensor, output_mat, input_mat);
}

void default_transformers(ImageTensor *image_tensor,
                          cv::Mat &output_mat,
                          cv::Mat &input_mat) {
  cv::resize(input_mat, output_mat, output_mat.size(), 0, 0);
}

transformers_func get_transformers(const std::string &transformer) {
  static std::unordered_map<std::string, transformers_func> transformers_map = {
      {"mobilenetv1_transformers", mobilenetv1_transformers},
      {"mobilenetv2_transformers", mobilenetv2_transformers},
      {"resnet18_transformers", resnet18_transformers},
      {"resnet50_transformers", resnet50_transformers},
      {"googlenet_transformers", googlenet_transformers},
      {"efficientnet_transformers", efficientnet_transformers},
      {"yolov2_transformers", yolov2_transformers},
      {"yolov3_transformers", yolov3_transformers},
      {"yolov5_transformers", yolov5_transformers},
      {"default_transformers", default_transformers}};
  return transformers_map[transformer];
}
