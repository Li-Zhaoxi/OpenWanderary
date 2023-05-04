// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/method_data.h"

#include "utils/tensor_utils.h"

TensorVector::TensorVector(std::vector<hbDNNTensorProperties> &properties) {
  tensors.resize(properties.size());
  for (int i = 0; i < properties.size(); ++i) {
    hbDNNTensorProperties &p = properties[i];
    int out_aligned_size = p.alignedByteSize;
    tensors[i].properties = p;
    hbSysMem &mem = tensors[i].sysMem[0];
    LOG_IF(FATAL, hbSysAllocCachedMem(&mem, out_aligned_size))
        << "Allocate system memory failed";
  }
}

TensorVector::~TensorVector() {
  release_output_tensor(tensors);
  tensors.clear();
}
