#pragma once

#include <set>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <wanderary/process/process_utils.h>
#include <wanderary/utils/class_registry.h>

#include <opencv2/opencv.hpp>

namespace wdr::proc {

struct ProcessRecorder {
  std::optional<ImageAffineParms> affine{std::nullopt};
};

class ProcessBase {
 public:
  explicit ProcessBase(const std::string &name);
  virtual ~ProcessBase() = default;

  static void make_active();

  virtual void Forward(cv::Mat *data,
                       ProcessRecorder *recorder = nullptr) const;
  const std::string &name() const { return name_; }

 private:
  const std::string name_;
};

class ProcessManager {
 public:
  explicit ProcessManager(const utils::json &cfg);
  void Forward(cv::Mat *data, ProcessRecorder *recorder = nullptr) const;

  static std::set<std::string> RegisteredNames();

 private:
  std::vector<std::unique_ptr<ProcessBase>> processes_;
  std::string manger_name_;
};

}  // namespace wdr::proc
