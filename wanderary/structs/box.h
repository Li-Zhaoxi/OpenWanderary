#pragma once
#include <string>
#include <vector>

#include <wanderary/utils/enum_traits.h>
#include <wanderary/utils/json_utils.h>

#include <opencv2/opencv.hpp>

ENUM_NUMBERED_REGISTER(Label2DSource,  //
                                       // 自由标准，无法从id推断出具体label
                       ((kCustom, 0, "custom"))  //
                       ((kWaymo, 1, "waymo"))    //
)
ENUM_CONVERSION_REGISTER(Label2DSource, Label2DSource::kCustom, "custom")

ENUM_NUMBERED_REGISTER(DifficultyLevel,  //
                                         // 自由标准，无法从id推断出具体label
                       ((kUnknown, 0, "unknown"))  //
                       ((kLevel1, 1, "level1"))    //
                       ((kLevel2, 2, "level2"))    //
)
ENUM_CONVERSION_REGISTER(DifficultyLevel, DifficultyLevel::kUnknown, "unknown")

namespace wdr {

struct AnnoMeta {
  std::string id;
  DifficultyLevel detection_difficulty_level = DifficultyLevel::kUnknown;
  DifficultyLevel tracking_difficulty_level = DifficultyLevel::kUnknown;
  wdr::ordered_json dump() const;
  static AnnoMeta load(const wdr::json& data);
};

struct Box2D {
  double score = 0;
  double x_min = 0;
  double y_min = 0;
  double h = 0;
  double w = 0;

  struct Label {
    int id = -1;
    Label2DSource source = Label2DSource::kCustom;
    wdr::ordered_json dump() const;
    static Label load(const wdr::json& data);
  } label;

  wdr::ordered_json dump() const;

  static Box2D load(const wdr::json& data);

  cv::Rect toCvRect() const;

  // 从左上角开始顺时针获取四个角点
  std::vector<cv::Point2d> CornerPoints() const;

  struct Attributes {
    std::optional<AnnoMeta> anno_meta{std::nullopt};
  } meta;
};

}  // namespace wdr
