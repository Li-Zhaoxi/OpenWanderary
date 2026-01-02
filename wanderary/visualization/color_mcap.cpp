#include "wanderary/visualization/color_mcap.h"

#include <utility>
#include <vector>

#include <glog/logging.h>
namespace wdr::vis {

cv::Scalar GetTextColor(const cv::Scalar& bg_color) {
  const double luminance =
      0.2126 * bg_color[2] + 0.7152 * bg_color[1] + 0.0722 * bg_color[0];
  return luminance > 140 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
}

std::vector<cv::Scalar> GetColorMap(int num_colors) {
  CHECK_LE(num_colors, 255);
  cv::Mat color_intensity = cv::Mat::zeros(1, num_colors, CV_8UC1);
  const double scale = 255.0 / num_colors;
  uchar* data = color_intensity.ptr<uchar>();
  for (int i = 0; i < num_colors; i++) data[i] = static_cast<int>(i * scale);
  cv::Mat img_color;
  cv::applyColorMap(color_intensity, img_color, cv::COLORMAP_RAINBOW);
  std::vector<cv::Scalar> colors;
  const cv::Vec3b* img_data = img_color.ptr<cv::Vec3b>();
  for (int i = 0; i < num_colors; i++) {
    const auto& bgr = img_data[i];
    colors.push_back(cv::Scalar(bgr[0], bgr[1], bgr[2]));
  }

  return colors;
}

WaymoTypeColormap::WaymoTypeColormap() : IntegralColormap<int>() {
  // vehicle: red
  color_map_[1] = std::make_pair(cv::Scalar(0, 0, 255, 255), "veh");
  // ped: green
  color_map_[2] = std::make_pair(cv::Scalar(0, 255, 0, 255), "ped");
  // cyclist: blue
  color_map_[4] = std::make_pair(cv::Scalar(255, 0, 0, 255), "cyc");
}

}  // namespace wdr::vis
