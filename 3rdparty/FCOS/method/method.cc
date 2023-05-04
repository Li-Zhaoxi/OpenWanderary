// Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/method.h"

#include "method/method_data.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/algorithm.h"
#include "utils/nms.h"

int Method::Init(const std::string &config_file_path) {
  std::ifstream ifs(config_file_path.c_str());
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open config file " << config_file_path
                         << " failed";
    return -1;
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string contents(buffer.str());
  ifs.close();
  return this->InitFromJsonString(contents);
}
