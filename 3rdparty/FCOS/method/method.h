// Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _METHOD_METHOD_H_
#define _METHOD_METHOD_H_

#include <string>

/**
 * Method
 */
class Method {
 public:
  /**
   * Init post process from file
   * @param[in] config_file_path: config file path
   *        config file should be in the json format
   *        for example:  { }
   * @return 0 if success
   */
  virtual int Init(const std::string &config_file_path);

  /**
   * Init post process from json string
   * @param[in] config: config json string
   * @return 0 if success
   */
  virtual int InitFromJsonString(const std::string &config) = 0;

  virtual ~Method() = default;
};

#endif  // _METHOD_METHOD_H_
