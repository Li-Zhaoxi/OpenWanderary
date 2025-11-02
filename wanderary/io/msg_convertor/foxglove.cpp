#include "wanderary/io/msg_convertor/foxglove.h"

#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "wanderary/io/msg_convertor/protobuf.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/path.h"

namespace wdr::msg {

void ConvertImageToMsg(const std::string &image_path, int64_t timestamp,
                       const std::string &frame_id,
                       foxglove::CompressedImage *msg) {
  CHECK(wdr::path::exist(image_path))
      << "Image file does not exist: " << image_path;

  ConvertTimestampToMsg(timestamp, msg->mutable_timestamp());
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
    buf = wdr::ReadBytesFromFile<uchar>(image_path);
  }
  msg->set_data(buf.data(), buf.size());
}

int64_t ConvertImageFromMsg(const foxglove::CompressedImage &msg, bool decode,
                            cv::Mat *image, std::string *format) {
  const auto &fmt = msg.format();
  if (format != nullptr) *format = fmt;

  if (format == nullptr && decode == false)
    LOG(FATAL) << "format is not specified and decode is false";

  const auto &data = msg.data();
  cv::Mat tmp(1, data.size(), CV_8UC1);
  std::memcpy(tmp.data, data.data(), data.size());
  if (decode) {
    if (fmt == "png" || fmt == "jpeg") {
      *image = cv::imdecode(tmp, cv::IMREAD_UNCHANGED);
    } else {
      LOG(FATAL) << "unsupported image format: " << fmt;
    }
  } else {
    *image = tmp;
  }

  return ConvertTimestampFromMsg(msg.timestamp());
}

}  // namespace wdr::msg
