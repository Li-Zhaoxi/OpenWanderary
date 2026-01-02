#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

namespace wdr::vis {

/**
 * \brief 根据背景色计算最佳文字颜色(黑或白)
 * \param[in] bg_color 背景色
 * \return 文字颜色
 * \note 会根据背景色计算相对亮度 (ITU-R BT.709标准)
 */
cv::Scalar GetTextColor(const cv::Scalar& bg_color);

/**
 * \brief 获取每个类别的颜色
 * \param[in] num_classes 类别数量
 * \return 每个类别的颜色
 */
std::vector<cv::Scalar> GetColorMap(int num_classes);

// 抽象基类 - 使用泛型接口
template <typename InputType, typename Enable = void>
class IColorMap {
 public:
  virtual ~IColorMap() = default;

  // 主要接口：根据输入数据返回颜色
  virtual cv::Scalar getColor(const InputType& input,
                              std::string* text) const = 0;

  // 批量处理接口
  virtual std::vector<cv::Scalar> getColors(
      const std::vector<InputType>& inputs) const {
    const int num_inputs = inputs.size();
    std::vector<cv::Scalar> colors(num_inputs);
    for (int k = 0; k < num_inputs; k++)
      colors[k] = getColor(inputs[k], nullptr);
    return colors;
  }
};

template <typename T>
class IntegralColormap
    : public IColorMap<
          T, typename std::enable_if<std::is_integral<T>::value>::type> {
 public:
  IntegralColormap() = default;
  ~IntegralColormap() override = default;

  cv::Scalar getColor(const T& input, std::string* text) const override {
    cv::Scalar res(0, 0, 0, 255);
    auto iter = color_map_.find(input);
    if (iter != color_map_.end()) {
      res = iter->second.first;
      if (text) *text = iter->second.second;
    } else {
      LOG(FATAL) << "[" << __FUNCTION__ << "] No color for input " << input;
    }
    return res;
  }

 protected:
  std::map<T, std::pair<cv::Scalar, std::string>> color_map_;
};

class WaymoTypeColormap : public IntegralColormap<int> {
 public:
  WaymoTypeColormap();
  ~WaymoTypeColormap() override = default;
};

}  // namespace wdr::vis
