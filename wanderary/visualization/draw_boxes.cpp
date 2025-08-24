#include "wanderary/visualization/draw_boxes.h"

#include <vector>
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

}  // namespace wdr::vis
