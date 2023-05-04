// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _METHOD_METHOD_FACTORY_H_
#define _METHOD_METHOD_FACTORY_H_

#include <string>
#include <unordered_map>

#include "base/common_def.h"
#include "base/perception_common.h"
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include "input/input_data.h"
#include "method.h"

#define DEFINE_AND_REGISTER_METHOD(method_name)               \
  Method *method_name##_creator() { return new method_name; } \
  static MethodRegistry method_registry(#method_name, method_name##_creator);

typedef Method *(*MethodCreator)();

class MethodFactory {
 public:
  static MethodFactory *GetInstance() {
    static MethodFactory ins;
    return &ins;
  }

  Method *GetMethod(const std::string &method_name) {
    if (method_registry_.find(method_name) == method_registry_.end()) {
      VLOG(EXAMPLE_SYSTEM) << "method " << method_name
                           << " has not been registered.";
      return nullptr;
    }
    return method_registry_[method_name]();
  }

  int32_t MethodRegister(const std::string &method_name,
                         MethodCreator creator) {
    if (method_registry_.find(method_name) != method_registry_.end()) {
      VLOG(EXAMPLE_DEBUG) << "method " << method_name
                          << " has been registered.";
    }
    method_registry_[method_name] = creator;
    return 0;
  }

  std::unordered_map<std::string, MethodCreator> method_registry_;
};

class MethodRegistry {
 public:
  explicit MethodRegistry(const std::string &method_name,
                          MethodCreator creator) noexcept {
    MethodFactory::GetInstance()->MethodRegister(method_name, creator);
  }
};

#endif  // _METHOD_METHOD_FACTORY_H_
