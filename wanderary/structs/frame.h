#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <wanderary/structs/box.h>

#include <opencv2/opencv.hpp>

namespace wdr {

struct ImageFile {
  std::string rawpath;
  std::unique_ptr<cv::Mat> data{nullptr};

  static ImageFile create(const std::string& rawpath, bool load_data);
  ImageFile clone() const;
};

struct Instance {
  std::optional<Box2D> box2d;
};

struct Frame {
  int64_t start_timestamp = -1;
  std::vector<Instance> instances;

  Frame clone() const;
  struct Metas {
    std::optional<ImageFile> image_file;
    Metas clone() const;
  } meta;
};

}  // namespace wdr
