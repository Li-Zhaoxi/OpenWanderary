#include "wanderary/data_loader/data_loader.h"

#include <set>
#include <string>

#include <glog/logging.h>

#include "wanderary/utils/class_registry.h"

namespace wdr::loader {

std::set<std::string> DataLoader::RegisteredNames() {
  return ClassRegistry<BaseDataset>::RegisteredClassNames();
}

DataLoader::DataLoader(const json &cfg) {
  const auto dataset_name = wdr::GetData<std::string>(cfg, "name");
  const auto dataset_cfg = wdr::GetData<json>(cfg, "config");
  this->dataset_ =
      ClassRegistry<BaseDataset>::createInstance(dataset_name, dataset_cfg);
  LOG(INFO) << "Created process: " << this->dataset_->name();
  if (cfg.contains("input")) {
    this->dataset_->load(cfg["input"]);
  }
}

DataLoader::DataLoader(const std::string &dataset_name, const json &cfg) {
  this->dataset_ =
      ClassRegistry<BaseDataset>::createInstance(dataset_name, cfg);
  LOG(INFO) << "Created process: " << this->dataset_->name();
}

void DataLoader::load(const wdr::json &data) {
  DCHECK(this->dataset_ != nullptr);
  this->dataset_->load(data);
}

int DataLoader::size() const {
  DCHECK(this->dataset_ != nullptr);
  return this->dataset_->size();
}

Frame DataLoader::at(int idx) const {
  DCHECK(this->dataset_ != nullptr);
  return this->dataset_->at(idx);
}

}  // namespace wdr::loader
