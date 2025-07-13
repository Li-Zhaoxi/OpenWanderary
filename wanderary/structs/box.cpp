#include "wanderary/structs/box.h"

#include <string>

#include <glog/logging.h>

namespace wdr {

wdr::ordered_json Box2D::dump() const {
  wdr::ordered_json res;
  res["dtype"] = "Box2D";
  res["score"] = score;
  res["x_min"] = x_min;
  res["y_min"] = y_min;
  res["w"] = w;
  res["h"] = h;
  res["label_id"] = label_id;
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
  res.label_id = wdr::GetData<int>(data, "label_id");
  return res;
}

}  // namespace wdr
