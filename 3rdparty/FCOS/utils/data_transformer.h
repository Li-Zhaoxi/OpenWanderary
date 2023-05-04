// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _DATA_TRANSFORMER_H_
#define _DATA_TRANSFORMER_H_

#include <ostream>
#include <string>
#include <unordered_map>

#include "base/common_def.h"
#include "utils/image_utils.h"

typedef void (*transformers_func)(ImageTensor *, cv::Mat &, cv::Mat &);

/**
 * mobilenetv1 transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void mobilenetv1_transformers(ImageTensor *image_tensor,
                              cv::Mat &output_mat,
                              cv::Mat &input_mat);

/**
 * mobilenetv2 transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void mobilenetv2_transformers(ImageTensor *image_tensor,
                              cv::Mat &output_mat,
                              cv::Mat &input_mat);

/**
 * resnet18 transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void resnet18_transformers(ImageTensor *image_tensor,
                           cv::Mat &output_mat,
                           cv::Mat &input_mat);

/**
 * resnet50 transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void resnet50_transformers(ImageTensor *image_tensor,
                           cv::Mat &output_mat,
                           cv::Mat &input_mat);

/**
 * googlenet transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void googlenet_transformers(ImageTensor *image_tensor,
                            cv::Mat &output_mat,
                            cv::Mat &input_mat);

/**
 * mobilenet transformers
 * ShortSideResizeTransformer(short_size=256)
 * CenterCropTransformer(crop_size=224)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void efficientnet_transformers(ImageTensor *image_tensor,
                               cv::Mat &output_mat,
                               cv::Mat &input_mat);

/**
 * mobilenet transformers
 * PadResizeTransformer(target_size=input_shape)
 * @param[out] output_mat
 * @param[in] input_mat
 */
void yolov2_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat);

void yolov3_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat);

void yolov5_transformers(ImageTensor *image_tensor,
                         cv::Mat &output_mat,
                         cv::Mat &input_mat);

void default_transformers(ImageTensor *image_tensor,
                          cv::Mat &output_mat,
                          cv::Mat &input_mat);

transformers_func get_transformers(const std::string &transformer);

#endif  // _DATA_TRANSFORMER_H_
