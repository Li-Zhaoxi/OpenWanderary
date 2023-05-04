// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/tensor_utils.h"

#include <memory.h>

#include <iostream>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/image_utils.h"
#include "utils/utils.h"

void prepare_image_tensor(int height,
                          int width,
                          int channel,
                          hbDNNDataType image_data_type,
                          hbDNNTensorLayout layout,
                          hbDNNTensor *tensor) {
  auto &tensor_property = tensor->properties;
  tensor_property.tensorType = image_data_type;
  tensor_property.tensorLayout = layout;
  int h_idx, w_idx, c_idx;
  auto ret = get_tensor_hwc_index(tensor, &h_idx, &w_idx, &c_idx);
  if (0 != ret) {
    VLOG(EXAMPLE_SYSTEM) << "HB_BPU_getHWCIndex failed for image_data_type="
                         << image_data_type << "!";
  }
  tensor_property.validShape.numDimensions = 4;
  tensor_property.validShape.dimensionSize[0] = 1;
  tensor_property.validShape.dimensionSize[h_idx] = height;
  tensor_property.validShape.dimensionSize[w_idx] = width;
  tensor_property.validShape.dimensionSize[c_idx] = channel;
  tensor_property.alignedShape = tensor_property.validShape;
  if (image_data_type == HB_DNN_IMG_TYPE_Y) {
    tensor_property.validShape.dimensionSize[c_idx] = 1;
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    tensor_property.alignedShape.dimensionSize[w_idx] = stride;
    tensor_property.alignedShape.dimensionSize[c_idx] = 1;
    hbSysAllocCachedMem(&tensor->sysMem[0], height * stride);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12) {
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    tensor_property.alignedShape.dimensionSize[w_idx] = stride;
    hbSysAllocCachedMem(&tensor->sysMem[0], y_length + uv_length);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    tensor_property.alignedShape.dimensionSize[w_idx] = stride;
    hbSysAllocCachedMem(&tensor->sysMem[0], y_length);
    hbSysAllocCachedMem(&tensor->sysMem[1], uv_length);
  } else if (image_data_type == HB_DNN_IMG_TYPE_YUV444 ||
             image_data_type == HB_DNN_IMG_TYPE_BGR ||
             image_data_type == HB_DNN_IMG_TYPE_RGB ||
             image_data_type == HB_DNN_TENSOR_TYPE_S8) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * width * channel);
  } else if (image_data_type == HB_DNN_TENSOR_TYPE_F32) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * width * channel * 4);
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Unimplemented for data type:" << image_data_type;
  }
}

int read_image_tensor(std::string &path,
                      ImageTensor *image_tensor,
                      hbDNNTensor *tensor,
                      transformers_func transformers) {
  image_tensor->ori_image_path = path;
  image_tensor->image_name = get_file_name(path);
  auto data_type = tensor->properties.tensorType;
  int h_idx, w_idx, c_idx;
  get_tensor_hwc_index(tensor, &h_idx, &w_idx, &c_idx);

  auto height = tensor->properties.validShape.dimensionSize[h_idx];
  auto width = tensor->properties.validShape.dimensionSize[w_idx];
  auto stride = tensor->properties.alignedShape.dimensionSize[w_idx];

  int &ori_width = image_tensor->ori_image_width;
  int &ori_height = image_tensor->ori_image_height;

  // read and resize
  cv::Mat mat;
  cv::Mat bgr_mat = cv::imread(path);
  ori_width = bgr_mat.cols;
  ori_height = bgr_mat.rows;
  mat.create(height, width, bgr_mat.type());
  transformers(image_tensor, mat, bgr_mat);

  if (data_type == HB_DNN_IMG_TYPE_Y) {
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    uint8_t *data = gray.data;
    for (int h = 0; h < height; ++h) {
      auto *raw = data0 + h * stride;
      for (int w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
  } else if (data_type == HB_DNN_IMG_TYPE_NV12) {
    cv::Mat nv12;
    bgr_to_nv12(mat, nv12);
    uint8_t *data = nv12.data;

    // padding Y
    uint8_t *y_data = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    for (int32_t h = 0; h < height; ++h) {
      auto *raw = y_data + h * stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }

    // padding uv
    uint8_t *uv_data = y_data + height * stride;
    for (int32_t h = 0; h < height / 2; ++h) {
      auto *raw = uv_data + h * stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
  } else if (data_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    cv::Mat nv12;
    bgr_to_nv12(mat, nv12);
    uint8_t *data = nv12.data;

    // Copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; ++h) {
      auto *raw = y + h * stride;
      for (int w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
    // Copy uv data to data1
    uint8_t *uv = reinterpret_cast<uint8_t *>(tensor->sysMem[1].virAddr);
    memcpy(uv, nv12.data + height * width, height * width / 2);
  } else if (data_type == HB_DNN_IMG_TYPE_YUV444 ||
             data_type == HB_DNN_IMG_TYPE_BGR ||
             data_type == HB_DNN_IMG_TYPE_RGB) {
    cv::Mat convert_mat;
    int convert_code = data_type == HB_DNN_IMG_TYPE_YUV444 ? cv::COLOR_BGR2YUV
                                                           : cv::COLOR_BGR2RGB;

    if (data_type == HB_DNN_IMG_TYPE_BGR) {
      convert_mat = mat;
    } else {
      cv::cvtColor(mat, convert_mat, convert_code);
    }

    if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
      void *data = tensor->sysMem[0].virAddr;
      memcpy(data, convert_mat.ptr<uint8_t>(), height * width * 3);
    }
    if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
      int channel_size = height * width;
      uint8_t *mem = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
      nhwc_to_nchw(mem,
                   mem + channel_size,
                   mem + channel_size * 2,
                   convert_mat.ptr<uint8_t>(),
                   height,
                   width);
    }
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Un support model input data type: "
                         << data_type_enum_to_string(data_type);
    return -1;
  }

  flush_tensor(tensor);

  return 0;
}

void flush_tensor(hbDNNTensor *tensor) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  if (tensor->properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    hbSysFlushMem(&(tensor->sysMem[1]), HB_SYS_MEM_CACHE_CLEAN);
  }
}

void release_tensor(hbDNNTensor *tensor) {
  hbSysFreeMem(&(tensor->sysMem[0]));
  if (tensor->properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    hbSysFreeMem(&(tensor->sysMem[1]));
  }
}

void prepare_output_tensor(std::vector<hbDNNTensor> &output,
                           hbDNNHandle_t dnn_handle) {
  int out_num;
  hbDNNGetOutputCount(&out_num, dnn_handle);
  output.resize(out_num);
  for (int i = 0; i < out_num; ++i) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
    int out_aligned_size = output_properties.alignedByteSize;
    hbSysMem &mem = output[i].sysMem[0];
    hbSysAllocCachedMem(&mem, out_aligned_size);
  }
}

void release_output_tensor(std::vector<hbDNNTensor> &output) {
  for (auto tensor : output) {
    hbSysFreeMem(&(tensor.sysMem[0]));
  }
}

int32_t calculate_bin_data_length(int32_t height,
                                  int32_t width,
                                  hbDNNDataType data_type,
                                  int8_t channel) {
  int32_t channel_size = height * width;
  switch (data_type) {
    case HB_DNN_IMG_TYPE_Y:
      return channel_size;
    case HB_DNN_IMG_TYPE_NV12:
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
      return channel_size * channel / 2;
    case HB_DNN_IMG_TYPE_YUV444:
    case HB_DNN_IMG_TYPE_RGB:
    case HB_DNN_IMG_TYPE_BGR:
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
      return channel_size * channel;
    case HB_DNN_TENSOR_TYPE_F32:
      return channel_size * channel * 4;
      break;
    default:
      VLOG(EXAMPLE_SYSTEM) << "Unimplemented for data type:" << data_type;
      return -1;
  }
}

int32_t calculate_bin_data_length(std::vector<int> &dims,
                                  hbDNNDataType data_type) {
  int32_t length = 1;
  for (int32_t i = 0; i < dims.size(); i++) {
    length *= dims[i];
  }
  switch (data_type) {
    // TODO(@horizon.ai): check 4bit
    case HB_DNN_TENSOR_TYPE_S4:
    case HB_DNN_TENSOR_TYPE_U4:
      return length *= 0.5;
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
      return length *= 1;
    case HB_DNN_TENSOR_TYPE_F16:
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
      return length *= 2;
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
      return length *= 4;
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64:
    case HB_DNN_TENSOR_TYPE_U64:
      return length *= 8;
    default:
      VLOG(EXAMPLE_SYSTEM) << "Unimplemented for data type:" << data_type;
      return -1;
  }
}

void tensor_padding_y(hbDNNTensor *tensor, char *data) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto &aligned_shape = tensor_property.alignedShape;
  int32_t h_idx, w_idx, c_idx;
  auto ret = get_tensor_hwc_index(tensor, &h_idx, &w_idx, &c_idx);
  auto height = valid_shape.dimensionSize[h_idx];
  auto width = valid_shape.dimensionSize[w_idx];
  auto w_stride = aligned_shape.dimensionSize[w_idx];
  if (width == w_stride) {
    VLOG(EXAMPLE_DETAIL) << "do not need padding for y!";
    char *vir_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
    memcpy(vir_addr, data, height * width);
  } else {
    // padding Y
    uint8_t *y_data = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    uint8_t *image_data = reinterpret_cast<uint8_t *>(data);
    for (int32_t h = 0; h < height; ++h) {
      auto *raw = y_data + h * w_stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *image_data++;
      }
    }
  }
}

void tensor_padding_nv12(hbDNNTensor *tensor, char *data) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto &aligned_shape = tensor_property.alignedShape;
  int32_t h_idx, w_idx, c_idx;
  auto ret = get_tensor_hwc_index(tensor, &h_idx, &w_idx, &c_idx);
  auto height = valid_shape.dimensionSize[h_idx];
  auto width = valid_shape.dimensionSize[w_idx];
  auto w_stride = aligned_shape.dimensionSize[w_idx];
  if (width == w_stride) {
    VLOG(EXAMPLE_DETAIL) << "do not need padding for nv12!";
    char *vir_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
    memcpy(vir_addr, data, height * width * 3 / 2);
  } else {
    // padding Y
    uint8_t *y_data = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    uint8_t *image_data = reinterpret_cast<uint8_t *>(data);
    for (int32_t h = 0; h < height; ++h) {
      auto *raw = y_data + h * w_stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *image_data++;
      }
    }

    // padding uv
    uint8_t *uv_data = y_data + height * w_stride;
    for (int32_t h = 0; h < height / 2; ++h) {
      auto *raw = uv_data + h * w_stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *image_data++;
      }
    }
  }
}

void tensor_padding_nv12_sep(hbDNNTensor *tensor, char *data) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto &aligned_shape = tensor_property.alignedShape;
  int32_t h_idx, w_idx, c_idx;
  auto ret = get_tensor_hwc_index(tensor, &h_idx, &w_idx, &c_idx);
  auto height = valid_shape.dimensionSize[h_idx];
  auto width = valid_shape.dimensionSize[w_idx];
  auto w_stride = aligned_shape.dimensionSize[w_idx];
  if (width == w_stride) {
    VLOG(EXAMPLE_DETAIL) << "do not need padding for nv12 sep!";
    char *y_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
    int32_t y_length = height * width;
    memcpy(y_addr, data, y_length);
    char *uv_addr = reinterpret_cast<char *>(tensor->sysMem[1].virAddr);
    memcpy(uv_addr, data + y_length, y_length / 2);
  } else {
    // padding Y
    uint8_t *y_data = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    uint8_t *image_data = reinterpret_cast<uint8_t *>(data);
    for (int32_t h = 0; h < height; ++h) {
      auto *raw = y_data + h * w_stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *image_data++;
      }
    }

    // padding uv
    uint8_t *uv_data = reinterpret_cast<uint8_t *>(tensor->sysMem[1].virAddr);
    for (int32_t h = 0; h < height / 2; ++h) {
      auto *raw = uv_data + h * w_stride;
      for (int32_t w = 0; w < width; ++w) {
        *raw++ = *image_data++;
      }
    }
  }
}

void prepare_tensor_data(hbDNNTensor *tensor, char *data) {
  auto &tensor_property = tensor->properties;
  auto data_type = tensor_property.tensorType;
  switch (data_type) {
    case HB_DNN_IMG_TYPE_Y:
      tensor_padding_y(tensor, data);
      break;
    case HB_DNN_IMG_TYPE_NV12:
      tensor_padding_nv12(tensor, data);
      break;
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
      tensor_padding_nv12_sep(tensor, data);
      break;
    default:
      char *vir_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
      auto data_length = tensor->sysMem[0].memSize;
      memcpy(vir_addr, data, data_length);
  }
  flush_tensor(tensor);
}
