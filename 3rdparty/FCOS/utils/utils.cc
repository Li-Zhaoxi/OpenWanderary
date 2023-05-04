// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "glog/logging.h"

int read_binary_file(std::string &file_path, char **bin, int *length) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  *length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  *bin = new char[sizeof(char) * (*length)];
  ifs.read(*bin, *length);
  ifs.close();
  return 0;
}

int read_binary_file(std::string &file_path, char *bin) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  ifs.read(bin, length);
  ifs.close();
  return 0;
}

std::string data_type_enum_to_string(int32_t data_type) {
  switch (data_type) {
    case HB_DNN_IMG_TYPE_Y:
      return "HB_DNN_IMG_TYPE_Y";
    case HB_DNN_IMG_TYPE_NV12:
      return "HB_DNN_IMG_TYPE_YUV_NV12";
    case HB_DNN_IMG_TYPE_YUV444:
      return "HB_DNN_IMG_TYPE_YUV444";
    case HB_DNN_IMG_TYPE_BGR:
      return "HB_DNN_IMG_TYPE_BGR";
    case HB_DNN_IMG_TYPE_RGB:
      return "HB_DNN_IMG_TYPE_RGB";
    case HB_DNN_TENSOR_TYPE_U8:
      return "HB_DNN_TENSOR_TYPE_U8";
    case HB_DNN_TENSOR_TYPE_S8:
      return "HB_DNN_TENSOR_TYPE_S8";
    case HB_DNN_TENSOR_TYPE_F32:
      return "HB_DNN_TENSOR_TYPE_F32";
    case HB_DNN_TENSOR_TYPE_S32:
      return "HB_DNN_TENSOR_TYPE_S32";
    case HB_DNN_TENSOR_TYPE_U32:
      return "HB_DNN_TENSOR_TYPE_U32";
    case HB_DNN_TENSOR_TYPE_MAX:
    default:
      return "HB_DNN_TENSOR_TYPE_MAX";
  }
}

std::string layout_type_enum_to_string(int32_t layout) {
  switch (layout) {
    case HB_DNN_LAYOUT_NCHW:
      return "HB_DNN_LAYOUT_NCHW";
    case HB_DNN_LAYOUT_NHWC:
      return "HB_DNN_LAYOUT_NHWC";
    case HB_DNN_LAYOUT_NONE:
      return "HB_DNN_LAYOUT_NONE";
  }
  return "UNRECOGNIZED_LAYOUT";
}

std::string get_file_name(std::string &path) {
  int slash_pos = path.rfind('/');
  return path.substr(slash_pos + 1);
}

void split(std::string &str,
           char sep,
           std::vector<std::string> &tokens,
           int limit) {
  int pos = -1;
  while (true) {
    int next_pos = str.find(sep, pos + 1);
    if (next_pos == std::string::npos) {
      tokens.emplace_back(str.substr(pos + 1));
      break;
    }
    tokens.emplace_back(str.substr(pos + 1, next_pos - pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(next_pos + 1));
      break;
    }
    pos = next_pos;
  }
}

void rsplit(std::string &str,
            char sep,
            std::vector<std::string> &tokens,
            int limit) {
  int pos = str.size();
  while (true) {
    int prev_pos = str.rfind(sep, pos - 1);
    if (prev_pos == std::string::npos) {
      tokens.emplace_back(str.substr(0, pos));
      break;
    }

    tokens.emplace_back(str.substr(prev_pos + 1, pos - prev_pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(0, prev_pos));
      break;
    }

    pos = prev_pos;
  }
}

void nhwc_to_nchw(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  uint8_t *in_data,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = *(in_data++);
      *out_data1++ = *(in_data++);
      *out_data2++ = *(in_data++);
    }
  }
}

void nchw_to_nhwc(uint8_t *out_data,
                  uint8_t *in_data0,
                  uint8_t *in_data1,
                  uint8_t *in_data2,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data++ = *(in_data0++);
      *out_data++ = *(in_data1++);
      *out_data++ = *(in_data2++);
    }
  }
}

bool operator==(hbDNNTensorShape &lhs, hbDNNTensorShape &rhs) {
  if (lhs.numDimensions != rhs.numDimensions) return false;
  for (int i = 0; i < lhs.numDimensions; i++) {
    if (lhs.dimensionSize[i] != rhs.dimensionSize[i]) {
      return false;
    }
  }
  return true;
}

int get_tensor_hwc_index(hbDNNTensor *tensor,
                         int *h_index,
                         int *w_index,
                         int *c_index) {
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    *h_index = 1;
    *w_index = 2;
    *c_index = 3;
  } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    *c_index = 1;
    *h_index = 2;
    *w_index = 3;
  } else {
    return -1;
  }
  return 0;
}

int get_tensor_hw(hbDNNTensor &tensor, int *height, int *width) {
  int h_index = 0;
  int w_index = 0;
  if (tensor.properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
  } else if (tensor.properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    h_index = 2;
    w_index = 3;
  } else {
    return -1;
  }
  *height = tensor.properties.validShape.dimensionSize[h_index];
  *width = tensor.properties.validShape.dimensionSize[w_index];
  return 0;
}

int get_tensor_aligned_hw(hbDNNTensor &tensor, int *height, int *width) {
  int h_index = 0;
  int w_index = 0;
  if (tensor.properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
  } else if (tensor.properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    h_index = 2;
    w_index = 3;
  } else {
    return -1;
  }
  *height = tensor.properties.alignedShape.dimensionSize[h_index];
  *width = tensor.properties.alignedShape.dimensionSize[w_index];
  return 0;
}

float quanti_shift(int32_t data, uint32_t shift) {
  return static_cast<float>(data) / static_cast<float>(1 << shift);
}

float quanti_scale(int32_t data, float scale) { return data * scale; }

void initClsName(const std::string &cls_name_file,
                 std::vector<std::string> &cls_names) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    std::string line;
    while (std::getline(fi, line)) {
      cls_names.push_back(line);
    }
  } else {
    VLOG(EXAMPLE_SYSTEM) << "can not open cls name file";
  }
}

std::string json_to_string(rapidjson::Value &val) {
  rapidjson::StringBuffer buffer;
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  val.Accept(writer);
  return buffer.GetString();
}
