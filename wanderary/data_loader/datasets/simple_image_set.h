#pragma once

#include <string>
#include <vector>

#include <wanderary/data_loader/base_dataset.h>
#include <wanderary/utils/json_utils.h>

namespace wdr::loader {

class SimpleImageDataset : public BaseDataset {
 public:
  struct Config {
    std::string image_root;
    std::string name_list_path;  // txt格式，每行表示一个图像
  };
  explicit SimpleImageDataset(const json &cfg);

  int size() const override { return image_paths_.size(); }
  Frame at(int idx) const override;

 private:
  Config cfg_;
  std::vector<std::string> image_paths_;
};

}  // namespace wdr::loader
