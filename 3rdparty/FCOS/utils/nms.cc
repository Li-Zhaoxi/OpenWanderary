// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/nms.h"

#include <algorithm>
#include <vector>

#define NMS_MAX_INPUT (400)

void nms(std::vector<Detection> &input,
         float iou_threshold,
         int top_k,
         std::vector<Detection> &result,
         bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Detection>());
  if (input.size() > NMS_MAX_INPUT) {
    input.resize(NMS_MAX_INPUT);
  }

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    float width = input[i].bbox.xmax - input[i].bbox.xmin;
    float height = input[i].bbox.ymax - input[i].bbox.ymin;
    areas.push_back(width * height);
  }

  int count = 0;
  for (size_t i = 0; count < top_k && i < skip.size(); i++) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      if (suppress == false) {
        if (input[i].id != input[j].id) {
          continue;
        }
      }

      // intersection area
      float xx1 = std::max(input[i].bbox.xmin, input[j].bbox.xmin);
      float yy1 = std::max(input[i].bbox.ymin, input[j].bbox.ymin);
      float xx2 = std::min(input[i].bbox.xmax, input[j].bbox.xmax);
      float yy2 = std::min(input[i].bbox.ymax, input[j].bbox.ymax);

      if (xx2 > xx1 && yy2 > yy1) {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold) {
          skip[j] = true;
        }
      }
    }
    result.push_back(input[i]);
  }
}

void yolo5_nms(std::vector<Detection> &input,
               float iou_threshold,
               int top_k,
               std::vector<Detection> &result,
               bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Detection>());

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    float width = input[i].bbox.xmax - input[i].bbox.xmin;
    float height = input[i].bbox.ymax - input[i].bbox.ymin;
    areas.push_back(width * height);
  }

  int count = 0;
  for (size_t i = 0; count < top_k && i < skip.size(); i++) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      if (suppress == false) {
        if (input[i].id != input[j].id) {
          continue;
        }
      }

      // intersection area
      float xx1 = std::max(input[i].bbox.xmin, input[j].bbox.xmin);
      float yy1 = std::max(input[i].bbox.ymin, input[j].bbox.ymin);
      float xx2 = std::min(input[i].bbox.xmax, input[j].bbox.xmax);
      float yy2 = std::min(input[i].bbox.ymax, input[j].bbox.ymax);

      if (xx2 > xx1 && yy2 > yy1) {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold) {
          skip[j] = true;
        }
      }
    }
    result.push_back(input[i]);
  }
}
