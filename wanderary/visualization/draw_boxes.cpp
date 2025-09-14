#include "wanderary/visualization/draw_boxes.h"

#include <algorithm>
#include <string>
#include <vector>

#include "wanderary/visualization/color_mcap.h"
namespace wdr::vis {

void DrawBoxes2D(const std::vector<wdr::Box2D> &boxes, cv::Mat *img) {
  for (const auto &box : boxes) {
    cv::rectangle(*img, box.toCvRect(), cv::Scalar(0, 0, 255), 2);
    cv::rectangle(*img, cv::Point(box.x_min, box.y_min - 20),
                  cv::Point(box.x_min + box.w, box.y_min),
                  cv::Scalar(0, 0, 255), -1);
    cv::putText(*img, std::to_string(box.label_id),
                cv::Point(box.x_min, box.y_min - 10), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 0));
  }
}

Box2DDrawer::Box2DDrawer(int class_num,
                         const std::vector<std::string> &class_names) {
  class_names_ = class_names;
  class_num_ = class_num;
  if (class_names_.size() > 0) {
    CHECK_EQ(class_names_.size(), class_num_);
  }
  colors_ = GetColorMap(class_num);
}

void Box2DDrawer::draw(const std::vector<wdr::Box2D> &boxes, cv::Mat *img) {
  for (const auto &box2d : boxes) {
    CHECK_LT(box2d.label_id, class_num_);
    CHECK_GE(box2d.label_id, 0);
    const auto &color = colors_[box2d.label_id];

    // 绘制检测框
    cv::rectangle(*img, box2d.toCvRect(), color, 2);

    // 构建文本
    std::ostringstream oss;
    if (class_names_.size() > 0) {
      oss << class_names_[box2d.label_id] << ": " << std::fixed
          << std::setprecision(2) << box2d.score;
    } else {
      oss << box2d.label_id << ": " << std::fixed << std::setprecision(2)
          << box2d.score;
    }
    const std::string text = oss.str();

    // 绘制文本框
    int baseline = 0;
    const auto text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    baseline += 1;

    const auto bgcolor =
        cv::Scalar(std::max(color[0] - 50, 0.), std::max(color[1] - 50, 0.),
                   std::max(color[2] - 50, 0.));
    cv::rectangle(*img, cv::Point(box2d.x_min, box2d.y_min - text_size.height),
                  cv::Point(box2d.x_min + text_size.width, box2d.y_min),
                  bgcolor, -1);
    const auto font_color = GetTextColor(bgcolor);
    cv::putText(
        *img, text,
        cv::Point(box2d.x_min, box2d.y_min - text_size.height + baseline + 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1);
  }
}

}  // namespace wdr::vis
