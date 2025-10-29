#include "wanderary/io/msg_convertor/foxglove.h"

#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "wanderary/io/msg_convertor/protobuf.h"
#include "wanderary/utils/path.h"

namespace wdr::msg {

void ConvertMsgImage(const std::string &image_path, int64_t timestamp,
                     const std::string &frame_id,
                     foxglove::CompressedImage *msg) {
  CHECK(wdr::path::exist(image_path))
      << "Image file does not exist: " << image_path;

  ConvertTimestampMsg(timestamp, msg->mutable_timestamp());
  msg->set_frame_id(frame_id);

  const std::string ext = wdr::path::extname(image_path);

  bool need_read = true;
  if (ext == ".png") {
    msg->set_format("png");
    need_read = false;
  } else if (ext == ".jpg" || ext == ".jpeg") {
    msg->set_format("jpeg");
    need_read = false;
  } else if (ext == ".webp") {
    msg->set_format("webp");
    need_read = false;
  } else if (ext == ".avif") {
    msg->set_format("avif");
    need_read = false;
  } else {
    msg->set_format("png");
    need_read = true;
  }

  std::vector<uchar> buf;
  if (need_read) {
    const cv::Mat img = cv::imread(image_path);
    cv::imencode(".png", img, buf);

  } else {
    std::ifstream ifs(image_path, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    const size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    buf.resize(size);
    ifs.read(reinterpret_cast<char *>(buf.data()), size);
  }
  msg->set_data(buf.data(), buf.size());
}

}  // namespace wdr::msg
