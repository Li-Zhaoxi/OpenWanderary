#pragma once

#include <wanderary/process/process_base.h>
#include <wanderary/utils/enum_traits.h>
#include <wanderary/utils/json_utils.h>
namespace wdr::proc {
ENUM_NUMBERED_REGISTER(ImageFormatType,                 //
                       ((kUnknown, 0, "unknown"))       //
                       ((kResize, 1, "resize"))         //
                       ((kLetterBox, 2, "letter_box"))  //
                       ((kNone, 3, "none"))             //
)
ENUM_CONVERSION_REGISTER(ImageFormatType, ImageFormatType::kUnknown, "unknown")

struct FormatImageConfig {
  ImageFormatType type_ = ImageFormatType::kUnknown;
  int width_ = 0;
  int height_ = 0;
  bool cvt_nv12_ = false;
};

/**
 * \brief 格式化处理输入图像，按照指定的形式构造模型输入的图像
 * \note 配置参数:
 * \param type_ 图像格式化类型
 * \param width_ 输出的图像宽度
 * \param height_ 输出的图像高度
 * \param cvt_nv12_ 是否将图像转换为NV12格式
 */
class FormatImage : public ProcessBase {
 public:
  explicit FormatImage(const json &cfg);

  void Forward(cv::Mat *data,
               ProcessRecorder *recorder = nullptr) const override;

 private:
  FormatImageConfig cfg_;
};

}  // namespace wdr::proc
