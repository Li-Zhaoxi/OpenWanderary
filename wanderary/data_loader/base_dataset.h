#pragma once

#include <string>

#include <wanderary/structs/frame.h>
#include <wanderary/utils/json_utils.h>

namespace wdr::loader {

class BaseDataset {
 public:
  explicit BaseDataset(const std::string &name);
  virtual ~BaseDataset() = default;

  const std::string &name() const { return name_; }

  virtual void load(const wdr::json &data) = 0;
  virtual int size() const = 0;
  virtual Frame at(int idx) const = 0;

  static void make_active();

 private:
  const std::string name_;
};

}  // namespace wdr::loader
