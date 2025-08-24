#pragma once

#include <wanderary/utils/json_utils.h>

#include <opencv2/opencv.hpp>

namespace wdr {

struct Box2D {
  double score = 0;
  double x_min = 0;
  double y_min = 0;
  double h = 0;
  double w = 0;
  int label_id = -1;

  wdr::ordered_json dump() const;

  static Box2D load(const wdr::json& data);

  cv::Rect toCvRect() const;
};

}  // namespace wdr
