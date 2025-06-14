
#include "wanderary/process/processors/format_image.h"

#include <string>
#include <utility>

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

void FormatImage::Forward(cv::Mat *data, ProcessRecorder *recorder) const {
  DCHECK(data != nullptr);
  cv::Mat tmp;
  ImageAffineParms parms;

  switch (cfg_.type_) {
    case ImageFormatType::kNone:
      tmp = *data;
      parms = ImageAffineParms();
      break;
    case ImageFormatType::kResize:
      parms = ResizeImage(*data, cv::Size(cfg_.width_, cfg_.height_), &tmp);
      break;
    case ImageFormatType::kLetterBox:
      parms = LetterBoxImage(*data, cv::Size(cfg_.width_, cfg_.height_), &tmp);
      break;
    default:
      LOG(FATAL) << "Unsupported image format type: "
                 << ImageFormatType2str(cfg_.type_);
      break;
  }

  if (cfg_.cvt_nv12_)
    wdr::BGRToNV12(tmp, data);
  else
    *data = std::move(tmp);

  if (recorder) recorder->affine = std::move(parms);
}

REGISTER_DERIVED_CLASS(ProcessBase, FormatImage)

}  // namespace wdr::proc
