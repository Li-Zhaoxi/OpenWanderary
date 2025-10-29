#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "wanderary/io/mcap_writer.h"

using MCAPWriter = wdr::io::MCAPWriter;

TEST(MCAP, ImageIO) {
  std::vector<std::string> image_paths = {
      "../../test_data/tiny_coco/val/000000006818.jpg",
      "../../test_data/tiny_coco/val/000000017627.jpg"};

  MCAPWriter writer("test_image_io.mcap");
  for (const auto &image_path : image_paths) {
    writer.WriteImage("images", image_path);
  }
  writer.close();
}
