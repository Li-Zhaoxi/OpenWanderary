#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/data_loader/data_loader.h"
#include "wanderary/testing/data_checking.h"

using DataLoader = wdr::loader::DataLoader;

TEST(DataLoader, TestRegisterClass) {
  const auto reg_names = DataLoader::RegisteredNames();

  std::stringstream ss;
  for (const auto &name : reg_names) ss << name << ", ";
  LOG(INFO) << "Registered classes: " << reg_names.size()
            << ", names: " << ss.str();

  wdr::testing::Check({"SimpleImageDataset"}, reg_names);
}
