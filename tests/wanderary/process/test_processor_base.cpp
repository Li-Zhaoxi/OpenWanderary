#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/process/process_base.h"
#include "wanderary/testing/data_checking.h"

TEST(ProcessBase, TestRegisterClass) {
  const auto reg_names =
      ClassRegistry<wdr::proc::ProcessBase>::RegisteredClassNames();

  std::stringstream ss;
  for (const auto &name : reg_names) ss << name << ", ";
  LOG(INFO) << "Registered classes: " << reg_names.size()
            << ", names: " << ss.str();

  wdr::testing::Check({"FormatImage"}, reg_names);
}
