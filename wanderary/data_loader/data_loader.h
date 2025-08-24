#pragma once

#include <memory>
#include <set>
#include <string>

#include <wanderary/data_loader/base_dataset.h>
#include <wanderary/utils/json_utils.h>

namespace wdr::loader {

class DataLoader {
 public:
  explicit DataLoader(const json& cfg);
  DataLoader(const std::string& dataset_name, const json& cfg);

  static std::set<std::string> RegisteredNames();

  void load(const wdr::json& data);
  int size() const;
  Frame at(int idx) const;

 private:
  std::unique_ptr<BaseDataset> dataset_{nullptr};
};

}  // namespace wdr::loader
