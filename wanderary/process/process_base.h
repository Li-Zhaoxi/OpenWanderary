#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <wanderary/dnn/bpu_nets.h>
#include <wanderary/process/process_utils.h>
#include <wanderary/structs/box.h>
#include <wanderary/utils/class_registry.h>

#include <opencv2/opencv.hpp>

namespace wdr::proc {

struct ProcessRecorder {
  std::optional<ImageAffineParms> affine{std::nullopt};
  std::optional<wdr::dnn::DequantScales> dequant_scales{std::nullopt};
};

class ProcessBase {
 public:
  explicit ProcessBase(const std::string &name);
  virtual ~ProcessBase() = default;

  static void make_active();

  // 用于输入预处理
  virtual void Forward(cv::Mat *data,
                       ProcessRecorder *recorder = nullptr) const;

  // 用于输出特征后处理
  virtual void Forward(std::vector<cv::Mat> *feats,
                       std::vector<wdr::Box2D> *box2ds,
                       ProcessRecorder *recorder = nullptr) const;

  const std::string &name() const { return name_; }

 private:
  const std::string name_;
};

std::unique_ptr<ProcessBase> CreateProcessor(const json &cfg);

class ProcessManager {
 public:
  /**
   * \brief 定义一个ProcessManager
   * \param manager_name(str) 指定ProcessManager的名字
   * \param cfg(Json) 要求指定rocesses(List[Json])
   */
  ProcessManager(const std::string &manager_name, const json &cfg);

  /**
   * \brief 定义一个ProcessManager
   * \note 要求指定manager_name(str)和processes(List[Json])
   */
  explicit ProcessManager(const json &cfg);
  void Forward(cv::Mat *data, ProcessRecorder *recorder = nullptr) const;

  // 用于输出特征后处理
  void Forward(std::vector<cv::Mat> *feats, std::vector<wdr::Box2D> *box2ds,
               ProcessRecorder *recorder = nullptr) const;

  static std::set<std::string> RegisteredNames();

 private:
  std::vector<std::unique_ptr<ProcessBase>> processes_;
  std::string manager_name_;
};

}  // namespace wdr::proc
