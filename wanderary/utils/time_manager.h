#pragma once
#include <chrono>
#include <map>
#include <mutex>
#include <string>

namespace wdr {

class TimerManager {
 public:
  void start(const std::string& phase);

  void stop(const std::string& phase);

  void printStatistics(const std::string& phase) const;

  void printStatistics() const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::chrono::steady_clock::duration> durations;
  std::map<std::string, std::chrono::steady_clock::time_point> startTimes;
};

class AutoScopeTimer {
 public:
  AutoScopeTimer(const std::string& phase, TimerManager* mgr)
      : manager_(mgr), phase_(phase) {
    manager_->start(phase_);
  }

  ~AutoScopeTimer() { manager_->stop(phase_); }

 private:
  TimerManager* manager_{nullptr};
  const std::string phase_;
};

}  // namespace wdr
