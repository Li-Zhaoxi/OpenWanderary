#include "wanderary/structs/box.h"

#include <string>
#include <utility>

#include <glog/logging.h>

namespace wdr {

wdr::ordered_json AnnoMeta::dump() const {
  wdr::ordered_json data;
  data["id"] = id;
  data["detection_difficulty_level"] =
      DifficultyLevel2str(detection_difficulty_level);
  data["tracking_difficulty_level"] =
      DifficultyLevel2str(tracking_difficulty_level);
  return data;
}

AnnoMeta AnnoMeta::load(const wdr::json& data) {
  AnnoMeta res;
  res.id = wdr::GetData<std::string>(data, "id");
  res.detection_difficulty_level = str2DifficultyLevel(
      wdr::GetData<std::string>(data, "detection_difficulty_level"));
  res.tracking_difficulty_level = str2DifficultyLevel(
      wdr::GetData<std::string>(data, "tracking_difficulty_level"));
  return res;
}

wdr::ordered_json Box2D::Label::dump() const {
  wdr::ordered_json res;
  res["id"] = id;
  res["source"] = Label2DSource2str(source);
  return res;
}

Box2D::Label Box2D::Label::load(const wdr::json& data) {
  Box2D::Label res;
  res.id = wdr::GetData<int>(data, "id");
  res.source = str2Label2DSource(wdr::GetData<std::string>(data, "source"));
  return res;
}

wdr::ordered_json Box2D::dump() const {
  wdr::ordered_json res;
  res["dtype"] = "Box2D";
  res["score"] = score;
  res["x_min"] = x_min;
  res["y_min"] = y_min;
  res["w"] = w;
  res["h"] = h;
  res["label"] = this->label.dump();

  wdr::json meta;
  if (this->meta.anno_meta.has_value())
    meta["anno_meta"] = this->meta.anno_meta->dump();
  res["meta"] = std::move(meta);

  return res;
}

Box2D Box2D::load(const wdr::json& data) {
  Box2D res;
  const auto dtype = wdr::GetData<std::string>(data, "dtype");
  CHECK_EQ(dtype, "Box2D");

  res.score = wdr::GetData<double>(data, "score");
  res.x_min = wdr::GetData<double>(data, "x_min");
  res.y_min = wdr::GetData<double>(data, "y_min");
  res.w = wdr::GetData<double>(data, "w");
  res.h = wdr::GetData<double>(data, "h");
  res.label = Box2D::Label::load(wdr::GetData<wdr::json>(data, "label"));

  wdr::json meta = wdr::GetData<wdr::json>(data, "meta");
  for (const auto& [k, v] : meta.items()) {
    if (k == "anno_meta") res.meta.anno_meta = AnnoMeta::load(v);
  }

  return res;
}

cv::Rect Box2D::toCvRect() const { return cv::Rect(x_min, y_min, w, h); }

}  // namespace wdr
