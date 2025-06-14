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

class FormatImage : public ProcessBase {
 public:
  explicit FormatImage(const utils::json &cfg);

  void Forward(const cv::Mat &input, cv::Mat *output,
               ProcessRecorder *recorder = nullptr) const override;

 private:
  FormatImageConfig cfg_;
};

}  // namespace wdr::proc
