#include "wanderary/io/msg_convertor/foxglove.h"

#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "wanderary/io/msg_convertor/protobuf.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/path.h"
#include "wanderary/visualization/color_mcap.h"

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

void ConvertImageToMsg(const ImageData &image_data, int64_t timestamp,
                       const std::string &frame_id,
                       foxglove::CompressedImage *msg) {
  ConvertTimestampToMsg(timestamp, msg->mutable_timestamp());
  msg->set_frame_id(frame_id);

  std::vector<uchar> buf;
  switch (image_data.type) {
    case ImageDataType::kRaw: {
      msg->set_format("raw");
      cv::imencode(".png", image_data.data, buf);
      break;
    }
    case ImageDataType::kJpeg: {
      msg->set_format("jpeg");
      buf.resize(image_data.data.cols);
      memcpy(buf.data(), image_data.data.data, image_data.data.cols);
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported image data type: "
                 << ImageDataType2str(image_data.type);
      break;
    }
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

void ConvertPoint2dToMsg(const cv::Point2d &pt2d, foxglove::Point2 *msg) {
  msg->set_x(pt2d.x);
  msg->set_y(pt2d.y);
}

void ConvertColorToMsg(const cv::Scalar &color, foxglove::Color *msg) {
  msg->set_r(color[2] / 255.0);
  msg->set_g(color[1] / 255.0);
  msg->set_b(color[0] / 255.0);
  msg->set_a(color[3] / 255.0);
}

std::string ConvertBox2DToMsg(int64_t timestamp, const Box2D &box,
                              foxglove::PointsAnnotation *msg) {
  ConvertTimestampToMsg(timestamp, msg->mutable_timestamp());

  msg->set_type(foxglove::PointsAnnotation_Type_LINE_LOOP);

  const auto cors = box.CornerPoints();
  for (const auto &cor : cors) ConvertPoint2dToMsg(cor, msg->add_points());

  const auto &label = box.label;
  cv::Scalar color;
  std::string label_str;
  switch (label.source) {
    case Label2DSource::kWaymo:
      color = wdr::vis::WaymoTypeColormap().getColor(label.id, &label_str);
      break;
    default:
      LOG(FATAL) << "unsupported label source: "
                 << Label2DSource2str(label.source);
  }
  ConvertColorToMsg(color, msg->mutable_outline_color());

  msg->set_thickness(2);
  return label_str;
}

void ConvertBox2DLabelToTextMsg(int64_t timestamp, const Box2D &box,
                                const std::string &text,
                                foxglove::TextAnnotation *msg) {
  ConvertTimestampToMsg(timestamp, msg->mutable_timestamp());
  ConvertPoint2dToMsg(cv::Point2d(box.x_min, box.y_min),
                      msg->mutable_position());
  msg->set_text(text);
  msg->set_font_size(10);
  ConvertColorToMsg(cv::Scalar(0, 0, 0, 255), msg->mutable_text_color());
  ConvertColorToMsg(cv::Scalar(255, 255, 255, 125),
                    msg->mutable_background_color());
}

void ConvertBoxes2DToMsg(int64_t timestamp, const std::vector<Box2D> &boxes,
                         foxglove::ImageAnnotations *msg) {
  for (const auto &box : boxes) {
    std::string label_str =
        ConvertBox2DToMsg(timestamp, box, msg->add_points());
    std::stringstream ss;
    ss << label_str << ":" << static_cast<int>(box.score * 100 + 0.5) << "%";
    ConvertBox2DLabelToTextMsg(timestamp, box, ss.str(), msg->add_texts());
  }
}

}  // namespace wdr::msg
