
#include "wanderary/process/processors/format_image.h"

#include <string>

#include "wanderary/process/process_base.h"
#include "wanderary/utils/convertor.h"
#include "wanderary/utils/enum_traits.h"
#include "wanderary/utils/json_utils.h"

namespace wdr::proc {

FormatImage::FormatImage(const utils::json &cfg) : ProcessBase("FormatImage") {
  cfg_.type_ = str2ImageFormatType(utils::GetData<std::string>(cfg, "type"));
  cfg_.width_ = utils::GetData<int>(cfg, "width");
  cfg_.height_ = utils::GetData<int>(cfg, "height");
  cfg_.cvt_nv12_ = utils::GetData<bool>(cfg, "cvt_nv12");
}

void FormatImage::Forward(const cv::Mat &input, cv::Mat *output,
                          ProcessRecorder *recorder) const {
  DCHECK(output != nullptr && recorder != nullptr);
  cv::Mat tmp;
  cv::Mat &usage_mat = cfg_.cvt_nv12_ ? tmp : *output;

  switch (cfg_.type_) {
    case ImageFormatType::kNone:
      input.copyTo(usage_mat);
      recorder->affine = ImageAffineParms();
      break;
    case ImageFormatType::kResize:
      recorder->affine =
          ResizeImage(input, cv::Size(cfg_.width_, cfg_.height_), &usage_mat);
      break;
    case ImageFormatType::kLetterBox:
      recorder->affine = LetterBoxImage(
          input, cv::Size(cfg_.width_, cfg_.height_), &usage_mat);
    default:
      LOG(FATAL) << "Unsupported image format type: "
                 << ImageFormatType2str(cfg_.type_);
      break;
  }

  if (cfg_.cvt_nv12_) {
    wdr::BGRToNV12(usage_mat, output);
  }
}

REGISTER_DERIVED_CLASS(ProcessBase, FormatImage)

}  // namespace wdr::proc
