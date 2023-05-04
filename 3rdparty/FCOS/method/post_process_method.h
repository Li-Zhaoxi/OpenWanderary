// Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _METHOD_POST_PROCESS_METHOD_H_
#define _METHOD_POST_PROCESS_METHOD_H_

#include <string>

#include "method.h"
#include "method_data.h"

/**
 * Method for post processing
 */
class PostProcessMethod : public Method {
 public:
  virtual PerceptionPtr DoProcess(ImageTensor *image_tensor,
                                  TensorVectorPtr &output_tensor) = 0;

  ~PostProcessMethod() override = default;
};

#endif  // _METHOD_POST_PROCESS_METHOD_H_
