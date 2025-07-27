#include "wanderary/data_loader/base_dataset.h"

#include <string>
namespace wdr::loader {
BaseDataset::BaseDataset(const std::string &name) : name_(name) {}

void BaseDataset::make_active() {}
}  // namespace wdr::loader
