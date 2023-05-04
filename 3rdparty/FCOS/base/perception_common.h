// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _BASE_PERCEPTION_COMMON_H_
#define _BASE_PERCEPTION_COMMON_H_

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

typedef struct Anchor {
  float cx{0.0};
  float cy{0.0};
  float w{0.0};
  float h{0.0};
  Anchor(float cx, float cy, float w, float h) : cx(cx), cy(cy), w(w), h(h) {}

  friend std::ostream &operator<<(std::ostream &os, const Anchor &anchor) {
    os << "[" << anchor.cx << "," << anchor.cy << "," << anchor.w << ","
       << anchor.h << "]";
    return os;
  }
} Anchor;

/**
 * Bounding box definition
 */
typedef struct Bbox {
  float xmin{0.0};
  float ymin{0.0};
  float xmax{0.0};
  float ymax{0.0};

  Bbox() {}

  Bbox(float xmin, float ymin, float xmax, float ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.xmin << ","
       << bbox.ymin << "," << bbox.xmax << "," << bbox.ymax << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Bbox() {}
} Bbox;

typedef struct Detection {
  int id{0};
  float score{0.0};
  Bbox bbox;
  const char *class_name{nullptr};
  Detection() {}

  Detection(int id, float score, Bbox bbox)
      : id(id), score(score), bbox(bbox) {}

  Detection(int id, float score, Bbox bbox, const char *class_name)
      : id(id), score(score), bbox(bbox), class_name(class_name) {}

  friend bool operator>(const Detection &lhs, const Detection &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection &det) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("label")"
       << ":" << det.id << ","
       << R"("class_name")"
       << ":\"" << det.class_name << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Detection() {}
} Detection;

static bool greater(Detection det1, Detection det2) {
  return (det1.score >= det2.score);
}

typedef struct Bbox3D {
  // geometric center coordinates: (x, y, z)
  float x{0.0};
  float y{0.0};
  float z{0.0};
  // width
  float w{0.0};
  // length
  float l{0.0};
  // height
  float h{0.0};
  // yaw angle
  float r{0.0};

  Bbox3D() {}

  Bbox3D(float x, float y, float z, float w, float l, float h, float r)
      : x(x), y(y), z(z), w(w), l(l), h(h), r(r) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox3D &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.x << "," << bbox.y
       << "," << bbox.z << "," << bbox.w << bbox.l << "," << bbox.h << ","
       << bbox.r << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Bbox3D() {}
} Bbox3D;

typedef struct Detection3D {
  Bbox3D bbox;
  float score{0.0};  // score
  int dir_label{0};  // direction label
  int top_label;     // classification lable
  Detection3D() {}

  Detection3D(Bbox3D bbox, float score, int dir_label, int top_label)
      : bbox(bbox), score(score), dir_label(dir_label), top_label(top_label) {}

  friend bool operator>(const Detection3D &lhs, const Detection3D &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection3D &det) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("score")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("dir_label")"
       << ":" << det.dir_label << R"("top_label")"
       << ":" << det.top_label << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~Detection3D() {}
} Detection3D;

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0) {}

  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Classification &cls) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(5) << cls.score << ","
       << R"("label")"
       << ":" << cls.id << ","
       << R"("class_name")"
       << ":"
       << "\"" << cls.class_name << "\""
       << "}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Classification() {}
} Classification;

struct Parsing {
  std::vector<int8_t> seg;
  int32_t num_classes = 0;
  int32_t width = 0;
  int32_t height = 0;
};

struct Point {
  std::vector<float> point;
  int32_t width = 0;
  int32_t height = 0;
};

struct MaskResultInfo {
  int32_t width = 0;
  int32_t height = 0;
  float h_base = 0;
  float w_base = 0;
  std::vector<Detection> det_info;
  std::vector<float> mask_info;
};

struct Perception {
  // Perception data
  std::vector<Detection> det;
  std::vector<Detection3D> det3d;
  std::vector<Classification> cls;
  Parsing seg;
  Point pt;
  MaskResultInfo mask;
  float h_base = 1;
  float w_base = 1;

  // TODO(@horizon.ai): remove from here
  uint64_t infer_duration;
  uint64_t pp_duration;

  // Perception type
  enum {
    DET = (1 << 0),
    CLS = (1 << 1),
    SEG = (1 << 2),
    MASK = (1 << 3),
    POINT = (1 << 4),
    DET3D = (1 << 5),
  } type;

  friend std::ostream &operator<<(std::ostream &os, Perception &perception) {
    os << "[";
    if (perception.type == Perception::DET) {
      auto &detection = perception.det;
      for (int i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }

    } else if (perception.type == Perception::CLS) {
      auto &cls = perception.cls;
      for (int i = 0; i < cls.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << cls[i];
      }
    } else if (perception.type == Perception::SEG) {
      auto &seg = perception.seg;
      for (int i = 0; i < seg.seg.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << static_cast<int>(seg.seg[i]);
      }
    } else if (perception.type == Perception::MASK) {
      auto &detection = perception.mask.det_info;
      for (int i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }
    } else if (perception.type == Perception::POINT) {
      auto &points = perception.pt.point;
      for (int i = 0; i < points.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << points[i];
      }
    } else if (perception.type == Perception::DET3D) {
      auto &rlts = perception.det3d;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << rlts[i];
      }
    }

    os << "]";
    return os;
  }
};

#endif  // _BASE_PERCEPTION_COMMON_H_
