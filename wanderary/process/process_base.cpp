#include "wanderary/process/process_base.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "wanderary/utils/time_manager.h"

using AutoScopeTimer = wdr::AutoScopeTimer;

namespace wdr::proc {

ProcessBase::ProcessBase(const std::string &name) : name_(name) {}

void ProcessBase::Forward(cv::Mat *data, ProcessRecorder *recorder) const {
  LOG(FATAL) << "Not implemented";
}

void ProcessBase::Forward(std::vector<cv::Mat> *feats,
                          std::vector<wdr::Box2D> *box2ds,
                          ProcessRecorder *recorder) const {
  LOG(FATAL) << "Not implemented";
}

void ProcessBase::make_active() {}

std::unique_ptr<ProcessBase> CreateProcessor(const json &cfg) {
  const auto proc_name = wdr::GetData<std::string>(cfg, "name");
  const auto proc_cfg = wdr::GetData<json>(cfg, "config");
  auto res = ClassRegistry<ProcessBase>::createInstance(proc_name, proc_cfg);
  LOG(INFO) << "Created process: " << res->name();
  return res;
}

ProcessManager::ProcessManager(const json &cfg) {
  manger_name_ = wdr::GetData<std::string>(cfg, "manger_name");
  const auto process_cfgs = wdr::GetData<std::vector<json>>(cfg, "processes");
  for (const auto &process_cfg : process_cfgs) {
    processes_.push_back(CreateProcessor(process_cfg));
    LOG(INFO) << "Add process: " << processes_.back()->name()
              << " to manager: " << manger_name_;
  }
}

void ProcessManager::Forward(cv::Mat *data, ProcessRecorder *recorder) const {
  for (const auto &process : processes_) {
    const std::string proc_phase = this->manger_name_ + "/" + process->name();
    {
      AutoScopeTimer scope_timer(proc_phase, &wdr::GlobalTimerManager());
      process->Forward(data, recorder);
    }
    wdr::GlobalTimerManager().printStatistics(proc_phase);
  }
}

void ProcessManager::Forward(std::vector<cv::Mat> *feats,
                             std::vector<wdr::Box2D> *box2ds,
                             ProcessRecorder *recorder) const {
  for (const auto &process : processes_) {
    const std::string proc_phase = this->manger_name_ + "/" + process->name();
    {
      AutoScopeTimer scope_timer(proc_phase, &wdr::GlobalTimerManager());
      process->Forward(feats, box2ds, recorder);
    }
    wdr::GlobalTimerManager().printStatistics(proc_phase);
  }
}

std::set<std::string> ProcessManager::RegisteredNames() {
  return ClassRegistry<wdr::proc::ProcessBase>::RegisteredClassNames();
}

}  // namespace wdr::proc
