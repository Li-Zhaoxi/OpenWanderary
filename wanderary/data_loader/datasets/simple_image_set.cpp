#include "wanderary/data_loader/datasets/simple_image_set.h"

#include <string>
#include <utility>

#include <glog/logging.h>

#include "wanderary/utils/class_registry.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/path.h"

namespace wdr::loader {

SimpleImageDataset::SimpleImageDataset(const json &cfg)
    : BaseDataset("SimpleImageDataset") {
  cfg_.image_root = wdr::GetData<std::string>(cfg, "image_root");
  cfg_.name_list_path = wdr::GetData<std::string>(cfg, "name_list_path");

  const auto name_list = wdr::ReadLinesFromFile(cfg_.name_list_path);
  for (const auto &name : name_list) {
    auto image_path = wdr::path::join({cfg_.image_root, name});
    CHECK(wdr::path::exist(image_path))
        << "Image file not found: " << image_path;
    image_paths_.push_back(std::move(image_path));
  }
}

Frame SimpleImageDataset::at(int idx) const {
  CHECK_LT(idx, image_paths_.size()) << "Index out of range";
  Frame frame;

  frame.meta.image_file =
      ImageFile::create(image_paths_[idx], /*load_data = */ true);

  return frame;
}

REGISTER_DERIVED_CLASS(BaseDataset, SimpleImageDataset)

}  // namespace wdr::loader
