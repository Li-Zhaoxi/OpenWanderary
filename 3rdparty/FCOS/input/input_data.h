// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_INPUT_DATA_H_
#define _INPUT_INPUT_DATA_H_

#include <ostream>
#include <string>

#include "base/common_def.h"
#include "dnn/hb_dnn.h"
#include "glog/logging.h"
#include "utils/utils.h"

typedef struct ImageTensor {
  hbDNNTensor tensor;
  int32_t frame_id = 0;
  std::string image_name;
  int ori_image_width;
  int ori_image_height;
  std::string ori_image_path;

  // TODO(@horizon.ai):
  bool is_pad_resize = false;

  inline int32_t height() {
    int height, width;

    if (0 != get_tensor_hw(tensor, &height, &width)) {
      VLOG(EXAMPLE_SYSTEM) << "get_tensor_hw failed";
    }
    return height;
  }

  inline int32_t width() {
    int height, width;
    if (0 != get_tensor_hw(tensor, &height, &width)) {
      VLOG(EXAMPLE_SYSTEM) << "get_tensor_hw failed";
    }
    return width;
  }

  inline int32_t ori_height() const { return ori_image_height; }

  inline int32_t ori_width() const { return ori_image_width; }

  friend std::ostream &operator<<(std::ostream &os, ImageTensor &image_tensor) {
    os << "{"
       << R"("image_name")"
       << ":\"" << image_tensor.image_name << "\", "
       << R"("image_width")"
       << ":" << image_tensor.ori_image_width << ", "
       << R"("image_height")"
       << ":" << image_tensor.ori_image_height << "}";
    return os;
  }
} ImageTensor;

#endif  // _INPUT_INPUT_DATA_H_
