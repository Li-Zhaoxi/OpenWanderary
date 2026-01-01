#include "wanderary/io/msg_convertor/open_waymo_dataset.h"

#include <memory>
#include <utility>

#include <glog/logging.h>

namespace wdr::msg {

namespace {

using WaymoCameraID = waymo::open_dataset::CameraName_Name;

SensorNameID ConvertCameraNameToWdrID(WaymoCameraID name) {
  SensorNameID res = SensorNameID::kUnknown;

  switch (name) {
    case WaymoCameraID::CameraName_Name_UNKNOWN:
      res = SensorNameID::kUnknown;
      break;
    case WaymoCameraID::CameraName_Name_FRONT:
      res = SensorNameID::kCameraFront;
      break;
    case WaymoCameraID::CameraName_Name_FRONT_LEFT:
      res = SensorNameID::kCameraFrontLeft;
      break;
    case WaymoCameraID::CameraName_Name_FRONT_RIGHT:
      res = SensorNameID::kCameraFrontRight;
      break;
    case WaymoCameraID::CameraName_Name_SIDE_LEFT:
      res = SensorNameID::kCameraSideLeft;
      break;
    case WaymoCameraID::CameraName_Name_SIDE_RIGHT:
      res = SensorNameID::kCameraSideRight;
      break;
    case WaymoCameraID::CameraName_Name_REAR_LEFT:
      res = SensorNameID::kCameraRearLeft;
      break;
    case WaymoCameraID::CameraName_Name_REAR:
      res = SensorNameID::kCameraRear;
      break;
    case WaymoCameraID::CameraName_Name_REAR_RIGHT:
      res = SensorNameID::kCameraRearRight;
      break;
    default:
      LOG(FATAL) << "Unknown sensor name: " << name;
      break;
  }

  return res;
}

}  // namespace

void ConvertCameraImageMsgToFrame(const waymo::open_dataset::CameraImage &frame,
                                  ImageFrame *frame2d) {
  // 加载原始数据
  ImageData image_data;
  const int image_size = frame.image().size();
  image_data.data.create(1, image_size, CV_8UC1);
  memcpy(image_data.data.data, frame.image().data(), image_size);
  image_data.type = ImageDataType::kJpeg;

  ImageFile image_file;
  image_file.data = std::make_unique<ImageData>(std::move(image_data));

  frame2d->meta.image_file = std::move(image_file);

  // 拷贝其他数据
  frame2d->sensor_name_id = ConvertCameraNameToWdrID(frame.name());
  frame2d->start_timestamp = frame.camera_trigger_time();
  frame2d->stop_timestamp = frame.camera_trigger_time();
}

void ConvertLabelBoxMsgToBox2D(const waymo::open_dataset::Label::Box &msg,
                               Box2D *box2d) {
  CHECK_NEAR(msg.center_z(), 0, 1e-6);
  CHECK_NEAR(msg.height(), 0, 1e-6);
  CHECK_NEAR(msg.heading(), 0, 1e-6);

  box2d->x_min = msg.center_x() - msg.length() / 2;
  box2d->y_min = msg.center_y() - msg.width() / 2;
  box2d->w = msg.length();
  box2d->h = msg.width();
}

void ConvertCameraLabelsMsgToFrame(
    const waymo::open_dataset::CameraLabels &frame, ImageFrame *frame2d) {
  CHECK(frame2d->sensor_name_id == ConvertCameraNameToWdrID(frame.name()));

  for (const auto &label : frame.labels()) {
    wdr::Box2D box2d;
    ConvertLabelBoxMsgToBox2D(label.box(), &box2d);

    box2d.label.id = label.type();
    box2d.label.source = Label2DSource::kWaymo;
    frame2d->boxes.push_back(std::move(box2d));
  }
}

void ConvertWaymoFrameMsgToMMFrame(const waymo::open_dataset::Frame &frame,
                                   MultiModalFrame *mmframe) {
  // 构造Image原始数据
  for (const auto &image : frame.images()) {
    ImageFrame frame2d;
    ConvertCameraImageMsgToFrame(image, &frame2d);
    mmframe->add(std::move(frame2d));
  }

  // 填充Box2D
  for (const auto &camera_label : frame.camera_labels()) {
    auto reframe = mmframe->mutable_camera_frame(
        ConvertCameraNameToWdrID(camera_label.name()));
    CHECK(reframe.has_value());
    ConvertCameraLabelsMsgToFrame(camera_label, &reframe->get());
  }
}

}  // namespace wdr::msg
