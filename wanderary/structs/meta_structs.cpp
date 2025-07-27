#include <memory>
#include <string>

#include <glog/logging.h>

#include "wanderary/structs/frame.h"
#include "wanderary/utils/path.h"

namespace wdr {

ImageFile ImageFile::create(const std::string& rawpath, bool load_data) {
  ImageFile res;
  res.rawpath = rawpath;
  if (load_data) {
    CHECK(wdr::path::exist(rawpath)) << "File does not exist: " << rawpath;
    res.data = std::make_unique<cv::Mat>(cv::imread(rawpath));
  }

  return res;
}

}  // namespace wdr
