#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/process/process_base.h"

TEST(ProcessBase, TestRegisterClass) {
  const auto reg_names =
      ClassRegistry<wdr::proc::ProcessBase>::RegisteredClassNames();
  LOG(INFO) << "Registered classes: " << reg_names.size();
  for (const auto& name : reg_names) {
    LOG(INFO) << "Registered class: " << name;
  }
}
