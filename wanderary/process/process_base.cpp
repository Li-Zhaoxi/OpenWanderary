#include "wanderary/process/process_base.h"

#include <string>
namespace wdr::proc {

ProcessBase::ProcessBase(const std::string &name) : name_(name) {}

void ProcessBase::Forward(const cv::Mat &input, cv::Mat *output,
                          ProcessRecorder *recorder) const {
  LOG(FATAL) << "Not implemented";
}

void ProcessBase::make_active() {}

ProcessManager::ProcessManager(const utils::json &cfg) {
  manger_name_ = utils::GetData<std::string>(cfg, "manger_name");
  const auto process_cfgs =
      utils::GetData<std::vector<utils::json>>(cfg, "processes");
  for (const auto &process_cfg : process_cfgs) {
    const auto proc_name = utils::GetData<std::string>(process_cfg, "name");
    const auto proc_cfg = utils::GetData<utils::json>(process_cfg, "config");
    processes_.push_back(
        ClassRegistry<ProcessBase>::createInstance(proc_name, proc_cfg));
    LOG(INFO) << "Add process: " << proc_name;
  }
}

}  // namespace wdr::proc
