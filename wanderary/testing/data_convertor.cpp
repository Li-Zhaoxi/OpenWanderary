#include "wanderary/testing/data_convertor.h"

#include <vector>

#include <glog/logging.h>

namespace wdr::testing {

std::vector<uchar> Convertor(const cv::Mat &img) {
  std::vector<uchar> data;
  const int byte_size = img.total() * img.elemSize();
  data.resize(byte_size);
  if (img.isContinuous()) {
    memcpy(data.data(), img.data, byte_size);
  } else {
    cv::Mat tmp;
    img.copyTo(tmp);
    memcpy(data.data(), tmp.data, byte_size);
  }
  return data;
}

}  // namespace wdr::testing
