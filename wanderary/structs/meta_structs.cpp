#include <memory>
#include <string>
#include <utility>

#include <glog/logging.h>

#include "wanderary/structs/frame.h"
#include "wanderary/utils/path.h"

namespace wdr {

ImageFile ImageFile::create(const std::string& rawpath, bool load_data) {
  ImageFile res;
  res.rawpath = rawpath;
  if (load_data) {
    CHECK(wdr::path::exist(rawpath)) << "File does not exist: " << rawpath;
    res.data = std::make_unique<ImageData>();
    res.data->data = cv::imread(rawpath);
    res.data->type = ImageDataType::kJpeg;
  } else {
    res.data = nullptr;
  }

  return res;
}

ImageFile ImageFile::clone() const {
  ImageFile res;
  res.rawpath = rawpath;
  if (data)
    res.data = std::make_unique<ImageData>(*data);
  else
    res.data = nullptr;
  return res;
}

}  // namespace wdr
