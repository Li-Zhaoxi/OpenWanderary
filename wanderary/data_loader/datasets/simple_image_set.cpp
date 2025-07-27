#include "wanderary/data_loader/datasets/simple_image_set.h"

#include <string>

#include <glog/logging.h>

#include "wanderary/utils/class_registry.h"

namespace wdr {

SimpleImageDataset::SimpleImageDataset(const json &cfg)
    : BaseDataset("SimpleImageDataset") {
  cfg_.list_path = wdr::GetData<std::string>(cfg, "list_path");
}

Frame SimpleImageDataset::at(int idx) const {
  CHECK_LT(idx, image_paths_.size()) << "Index out of range";
  Frame frame;

  frame.meta.image_file =
      ImageFile::create(image_paths_[idx], /*load_data = */ true);

  return frame;
}

REGISTER_DERIVED_CLASS(BaseDataset, SimpleImageDataset)

}  // namespace wdr
