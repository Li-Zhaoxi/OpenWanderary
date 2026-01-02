#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <wanderary/structs/box.h>
#include <wanderary/structs/enums.h>
#include <wanderary/utils/enum_traits.h>

#include <opencv2/opencv.hpp>

ENUM_NUMBERED_REGISTER(ImageDataType,              //
                       ((kUnknown, 0, "unknown"))  //
                       ((kRaw, 1, "raw"))          //
                       ((kJpeg, 2, "jped"))        //
)
ENUM_CONVERSION_REGISTER(ImageDataType, ImageDataType::kUnknown, "unknown")

namespace wdr {

struct ImageData {
  cv::Mat data;
  ImageDataType type = ImageDataType::kUnknown;
};

struct ImageFile {
  std::string rawpath;
  std::unique_ptr<ImageData> data{nullptr};

  ImageFile() = default;
  ImageFile(const ImageFile &other) {
    rawpath = other.rawpath;
    data = std::make_unique<ImageData>(*other.data);
  }
  ImageFile(ImageFile &&other) {
    rawpath = std::move(other.rawpath);
    data = std::move(other.data);
  }
  ImageFile &operator=(ImageFile &&other) {
    rawpath = std::move(other.rawpath);
    data = std::move(other.data);
    return *this;
  }

  static ImageFile create(const std::string &rawpath, bool load_data);
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

// Frame基础类
struct BaseFrame {
  BaseFrame() = default;
  virtual ~BaseFrame() = default;

  SensorNameID sensor_name_id = SensorNameID::kUnknown;
  int64_t start_timestamp = -1;
  int64_t stop_timestamp = -1;
};

// 图像帧
struct ImageFrame : public BaseFrame {
  ImageFrame() : BaseFrame() {}
  ~ImageFrame() override = default;

  std::vector<Box2D> boxes;
  struct Metas {
    std::optional<ImageFile> image_file;
  } meta;
};

// 多模态帧: 包含图像/Lidar等
class MultiModalFrame {
 public:
  void add(ImageFrame &&frame);
  std::optional<std::reference_wrapper<ImageFrame>> mutable_camera_frame(
      SensorNameID name_id);
  std::map<SensorNameID, std::reference_wrapper<const ImageFrame>>
  camera_frames() const;

 private:
  std::map<SensorNameID, ImageFrame> camera_frames_;  // 图像帧
};

}  // namespace wdr
