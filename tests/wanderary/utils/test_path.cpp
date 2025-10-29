#include <gtest/gtest.h>

#include "wanderary/utils/path.h"

TEST(PATH, extname) {
  EXPECT_EQ(wdr::path::extname("path/to/file.txt"), ".txt");
  EXPECT_EQ(wdr::path::extname("path/to/file.txt.json"), ".json");
  EXPECT_EQ(wdr::path::extname("path/to.png/file.txt.json"), ".json");
}
