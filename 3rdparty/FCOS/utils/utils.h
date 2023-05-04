// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_UTILS_H_
#define _UTILS_UTILS_H_

#include <string>
#include <vector>

#include "base/common_def.h"
#include "dnn/hb_dnn.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);

#define METHOD_CHECK_SUCCESS(value, errmsg, res)                     \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return res;                                                    \
    }                                                                \
  } while (0);

/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @param[out] length: bin length
 * @return 0 if success otherwise -1
 */
int read_binary_file(std::string &file_path, char **bin, int *length);

/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @return 0 if success otherwise -1
 */
int read_binary_file(std::string &file_path, char *bin);

/**
 * Parse data_type enum to string
 * @param[in] data_type enum
 * @return data_type string
 */
std::string data_type_enum_to_string(int32_t data_type);

/**
 * Layout string name
 * @param[in] layout
 * @return layout string
 */
std::string layout_type_enum_to_string(int32_t layout);

/**
 * Get filename
 * @param path: file path
 * @return filename
 */
std::string get_file_name(std::string &path);

/**
 * Split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void split(std::string &str,
           char sep,
           std::vector<std::string> &tokens,
           int limit = -1);

/**
 * Reverse split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void rsplit(std::string &str,
            char sep,
            std::vector<std::string> &tokens,
            int limit = -1);

/**
 * NHWC to NCHW
 * @param[out] out_data0: channel/planar 0 data
 * @param[out] out_data1: channel/planar 1 data
 * @param[out] out_data2: channel/planar 2 data
 * @param[in] in_data:
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nhwc_to_nchw(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  uint8_t *in_data,
                  int height,
                  int width);
/**
 * NCHW to NHWC
 * @param[out] out_data:
 * @param[in] in_data0: channel/planar 0 data
 * @param[in] in_data1: channel/planar 1 data
 * @param[in] in_data2: channel/planar 2 data
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nchw_to_nhwc(uint8_t *out_data,
                  uint8_t *in_data0,
                  uint8_t *in_data1,
                  uint8_t *in_data2,
                  int height,
                  int width);

/**
 * Test whether it's the same shape
 * @param[in] lhs:
 * @param[in] rhs:
 * @return true if the shape is equal
 */
bool operator==(hbDNNTensorShape &lhs, hbDNNTensorShape &rhs);

/**
 *
 * @param[in] tensor
 * @param[out] h_index
 * @param[out] w_index
 * @param[out] c_index
 * @return 0f if success
 */
int get_tensor_hwc_index(hbDNNTensor *tensor,
                         int *h_index,
                         int *w_index,
                         int *c_index);

int get_tensor_hw(hbDNNTensor &tensor, int *height, int *width);

/**
 *
 * @param tensor
 * @param height
 * @param width
 * @return
 */
int get_tensor_aligned_hw(hbDNNTensor &tensor, int *height, int *width);

/**
 * Quanti shift
 * @param[in] data
 * @param[in] shift
 * @return shift value
 */
float quanti_shift(int32_t data, uint32_t shift);

/**
 * Quanti scale
 * @param[in] data
 * @param[in] scale
 * @return scale value
 */
float quanti_scale(int32_t data, float scale);

/**
 * Read classification name list from config file
 * @param[in] cls_name_file classification name list file
 * @param[out] cls_names classification name list
 */
void initClsName(const std::string &cls_name_file,
                 std::vector<std::string> &cls_names);

/**
 * Dump rapidjson to string
 * @param[in] rapidjson::Value val
 * @return json string
 */
std::string json_to_string(rapidjson::Value &val);

#endif  // _UTILS_UTILS_H_
