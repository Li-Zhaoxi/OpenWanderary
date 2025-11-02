#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "wanderary/io/mcap_reader.h"
#include "wanderary/io/mcap_writer.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/utils/file_io.h"

using MCAPWriter = wdr::io::MCAPWriter;
using MCAPReader = wdr::io::MCAPReader;

TEST(MCAP, ImageIO) {
  std::vector<std::string> image_paths = {
      "../../test_data/tiny_coco/val/000000006818.jpg",
      "../../test_data/tiny_coco/val/000000017627.jpg"};

  MCAPWriter writer("test_image_io.mcap");
  const std::string image_topic = "images";
  for (const auto &image_path : image_paths) {
    writer.WriteImage(image_topic, image_path);
  }
  writer.close();

  MCAPReader reader("test_image_io.mcap");

  const int image_num = image_paths.size();
  ASSERT_EQ(reader.size(image_topic), image_num);

  for (int i = 0; i < image_num; ++i) {
    const auto &imgpath = image_paths[i];

    {  // 测试原始数据(不解码)
      auto buf = wdr::ReadBytesFromFile<uchar>(imgpath);
      cv::Mat data;
      reader.ReadImage(image_topic, i, &data, false);
      wdr::testing::CheckGeneralMat(cv::Mat(1, buf.size(), CV_8UC1, buf.data()),
                                    data, 1e-6);
    }

    {  // 测试解码后数据
      cv::Mat img = cv::imread(imgpath);
      cv::Mat data;
      reader.ReadImage(image_topic, i, &data, true);
      wdr::testing::CheckGeneralMat(img, data, 1e-6);
    }
  }

  reader.close();
}
