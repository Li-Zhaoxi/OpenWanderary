// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _METHOD_METHOD_DATA_H_
#define _METHOD_METHOD_DATA_H_

#include <fstream>
#include <memory>
#include <vector>

#include "base/perception_common.h"
#include "dnn/hb_dnn.h"
#include "input/input_data.h"

struct TensorVector {
  TensorVector() {}
  explicit TensorVector(std::vector<hbDNNTensorProperties> &properties);
  void Reset() {}
  ~TensorVector();

  std::vector<hbDNNTensor> tensors;
};

typedef std::shared_ptr<ImageTensor> ImageTensorPtr;
typedef std::shared_ptr<Perception> PerceptionPtr;
typedef std::shared_ptr<TensorVector> TensorVectorPtr;

#endif  // _METHOD_METHOD_DATA_H_
