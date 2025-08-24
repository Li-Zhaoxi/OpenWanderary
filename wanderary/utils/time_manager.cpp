#include "wanderary/utils/time_manager.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "wanderary/utils/common_utils.h"

namespace wdr {

void TimerManager::start(const std::string& phase) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(!wdr::contains(startTimes, phase));
  startTimes[phase] = std::chrono::steady_clock::now();
}

void TimerManager::stop(const std::string& phase) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(wdr::contains(startTimes, phase));
  const auto end = std::chrono::steady_clock::now();
  const auto duration = end - startTimes[phase];
  durations[phase] += duration;
  startTimes.erase(phase);
}

int TimerManager::getDuration(const std::string& phase) const {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(wdr::contains(durations, phase));
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      durations.at(phase));
  return ms.count();
}

void TimerManager::printStatistics(const std::string& phase) const {
  LOG(INFO) << "[" << phase << "]: " << getDuration(phase) / 1000.0 << "s";
}

void TimerManager::printStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [phase, duration] : durations) {
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    LOG(INFO) << "[" << phase << "]: " << ms.count() / 1000.0 << "s\n";
  }
}

void TimerManager::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  durations.clear();
  startTimes.clear();
}

std::set<std::string> TimerManager::getPhases() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::set<std::string> phases;
  for (const auto& [phase, _] : durations) phases.insert(phase);
  return phases;
}

void StatisticsTimeManager::add(const TimerManager& mgr) {
  const auto phases = mgr.getPhases();
  for (const auto& phase : phases) {
    const auto duration = mgr.getDuration(phase);
    if (wdr::contains(statistics_, phase)) {
      statistics_[phase].first += duration;
      statistics_[phase].second++;
    } else {
      statistics_[phase] = {duration, 1};
    }
  }
}

void StatisticsTimeManager::printStatistics() const {
  std::stringstream ss;
  ss << "Average statistics:\n";
  for (const auto& [phase, durations] : statistics_) {
    ss << "[" << phase
       << "]: " << static_cast<double>(durations.first) / durations.second
       << "ms, count: " << durations.second << "\n";
  }
  LOG(INFO) << ss.str();
}

TimerManager& GlobalTimerManager() {
  static TimerManager timerManager;
  return timerManager;
}

}  // namespace wdr
