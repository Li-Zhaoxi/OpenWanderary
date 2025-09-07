#pragma once
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wdr {

class TimerManager {
 public:
  TimerManager() = default;
  void start(const std::string& phase);

  void stop(const std::string& phase);

  void printStatistics(const std::string& phase) const;

  void printStatistics() const;

  int getDuration(const std::string& phase) const;

  std::set<std::string> getPhases() const;

  void reset();

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::chrono::steady_clock::duration> durations;
  std::map<std::string, std::chrono::steady_clock::time_point> startTimes;
};

class StatisticsTimeManager {
 public:
  StatisticsTimeManager() = default;
  void add(const TimerManager& mgr);
  void printStatistics() const;

 private:
  mutable std::mutex mtx_;
  std::map<std::string, std::pair<int, int>> statistics_;
};

class AutoScopeTimer {
 public:
  AutoScopeTimer(const std::string& phase, TimerManager* mgr)
      : manager_(mgr), phase_(phase) {
    if (manager_) manager_->start(phase_);
  }

  ~AutoScopeTimer() {
    if (manager_) manager_->stop(phase_);
  }

 private:
  TimerManager* manager_{nullptr};
  const std::string phase_;
};

TimerManager& GlobalTimerManager();

}  // namespace wdr
