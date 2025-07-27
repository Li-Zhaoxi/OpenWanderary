#pragma once

#include <memory>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

namespace wdr {

struct ImageFile {
  std::string rawpath;
  std::unique_ptr<cv::Mat> data{nullptr};

  static ImageFile create(const std::string& rawpath, bool load_data);
};

struct Frame {
  struct {
    std::optional<ImageFile> image_file;
  } meta;
};

}  // namespace wdr
