#include "wanderary/process/process_utils.h"

#include <algorithm>

namespace wdr::proc {

wdr::utils::json ImageAffineParms::dump() const {
  wdr::utils::json res;
  res["x_scale"] = x_scale;
  res["y_scale"] = y_scale;
  res["x_shift"] = x_shift;
  res["y_shift"] = y_shift;
  return res;
}

ImageAffineParms ResizeImage(const cv::Mat &img, const cv::Size &input_wh,
                             cv::Mat *out) {
  cv::resize(img, *out, input_wh, 0, 0, cv::INTER_NEAREST);

  ImageAffineParms parms;
  parms.x_scale =
      static_cast<double>(input_wh.width) / static_cast<double>(img.cols);
  parms.y_scale =
      static_cast<double>(input_wh.height) / static_cast<double>(img.rows);
  parms.x_shift = 0;
  parms.y_shift = 0;

  return parms;
}

ImageAffineParms LetterBoxImage(const cv::Mat &img, const cv::Size &input_wh,
                                cv::Mat *out) {
  ImageAffineParms parms;
  const double x_scale =
      static_cast<double>(input_wh.width) / static_cast<double>(img.cols);
  const double y_scale =
      static_cast<double>(input_wh.height) / static_cast<double>(img.rows);
  const double scale = std::min(x_scale, y_scale);
  parms.x_scale = scale;
  parms.y_scale = scale;

  const int new_w = static_cast<int>(img.cols * scale);
  parms.x_shift = (input_wh.width - new_w) / 2;
  const int x_other = input_wh.width - new_w - parms.x_shift;

  const int new_h = static_cast<int>(img.rows * scale);
  parms.y_shift = (input_wh.height - new_h) / 2;
  const int y_other = input_wh.height - new_h - parms.y_shift;

  cv::resize(img, *out, cv::Size(new_w, new_h));
  cv::copyMakeBorder(*out, *out, parms.y_shift, y_other, parms.x_shift, x_other,
                     cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

  return parms;
}

}  // namespace wdr::proc
