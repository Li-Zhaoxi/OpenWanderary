#include "wanderary/utils/time_manager.h"

#include <string>

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

TimerManager& GlobalTimerManager() {
  static TimerManager timerManager;
  return timerManager;
}

}  // namespace wdr
