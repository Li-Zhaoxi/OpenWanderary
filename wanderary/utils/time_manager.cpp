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
}

void TimerManager::printStatistics(const std::string& phase) const {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(wdr::contains(durations, phase));
  const auto ms =
      std::chrono::duration_cast<std::chrono::seconds>(durations.at(phase));
  LOG(INFO) << "[" << phase << "]: " << ms.count() << "s";
}

void TimerManager::printStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [phase, duration] : durations) {
    auto ms = std::chrono::duration_cast<std::chrono::seconds>(duration);
    LOG(INFO) << "[" << phase << "]: " << ms.count() << "s\n";
  }
}

}  // namespace wdr
