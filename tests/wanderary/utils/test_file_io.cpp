#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/utils/file_io.h"

TEST(FILEIO, ReadLines) {
  const std::string filepath = "../../test_data/tiny_coco/val_names.txt";

  const auto alllines = wdr::ReadLinesFromFile(filepath);
  LOG(INFO) << "First line: " << alllines[0];
  EXPECT_EQ(alllines.size(), 50);
}
