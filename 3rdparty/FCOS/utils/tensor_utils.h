// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_TENSOR_UTILS_H_
#define _UTILS_TENSOR_UTILS_H_

#include <string>
#include <vector>

#include "dnn/hb_dnn.h"
#include "input/input_data.h"
#include "utils/data_transformer.h"

/**
 * Align by 16
 */

#define ALIGN_16(v) ((v + (16 - 1)) / 16 * 16)

/**
 * Prepare image tensor
 * @param[in] height
 * @param[in] width
 * @param[in] data_type: tensor data type
 * @param[in] layout: tensor layout
 * @param[out] tensor
 */
void prepare_image_tensor(int height,
                          int width,
                          int channel,
                          hbDNNDataType data_type,
                          hbDNNTensorLayout layout,
                          hbDNNTensor *tensor);

/**
 * read image tensor
 * @param[in] path:
 * @param[out] image_tensor
 * @param[out] tensor:
 * @param[in] transformers:
 * @return 0 if success
 */
int read_image_tensor(std::string &path,
                      ImageTensor *image_tensor,
                      hbDNNTensor *tensor,
                      transformers_func transformers);

/**
 * Flush tensor
 * @param[in] tensor: Tensor to be flushed
 */
void flush_tensor(hbDNNTensor *tensor);

/**
 * Free tensor
 * @param tensor: Tensor to be released
 */
void release_tensor(hbDNNTensor *tensor);

/**
 * Prepare output tensor
 * @param output
 * @param model
 */
void prepare_output_tensor(std::vector<hbDNNTensor> &output,
                           hbDNNHandle_t dnn_handle);

/**
 * Release output tensor
 * @param output
 */
void release_output_tensor(std::vector<hbDNNTensor> &output);

/**
 * calculate input bin file length cross input height, width and datatype.
 * @param[in] height: input height
 * @param[in] width: input width
 * @param[in] data_type: input datatype
 * @return Natural number if success, -1 if failed
 */
int32_t calculate_bin_data_length(int32_t height,
                                  int32_t width,
                                  hbDNNDataType data_type,
                                  int8_t channel);

/**
 * calculate input bin file length cross input height, width and datatype.
 * @param[in] dim: input valid shape
 * @param[in] data_type: input datatype
 * @return Natural number if success, -1 if failed
 */
int32_t calculate_bin_data_length(std::vector<int> &dims,
                                  hbDNNDataType data_type);

/**
 * tensor padding for y
 * @param[in] tensor: tensor without data
 * @param[in] data: input data
 */
void tensor_padding_y(hbDNNTensor *tensor, char *data);

/**
 * tensor padding for nv12
 * @param[in] tensor: tensor without data
 * @param[in] data: input data
 */
void tensor_padding_nv12(hbDNNTensor *tensor, char *data);

/**
 * tensor padding for nv12 sep
 * @param[in] tensor: tensor without data
 * @param[in] data: input data
 */
void tensor_padding_nv12_sep(hbDNNTensor *tensor, char *data);

/**
 * padding for tensor
 * @param[in] tensor: tensor without data
 * @param[in] data: input data
 */
void prepare_tensor_data(hbDNNTensor *tensor, char *data);
#endif  // _UTILS_TENSOR_UTILS_H_
