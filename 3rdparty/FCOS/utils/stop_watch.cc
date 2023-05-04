// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/stop_watch.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

u_int64_t Stopwatch::CurrentTs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

Stopwatch::Stopwatch()
    : timing_count_(0),
      start_(std::chrono::steady_clock::now()),
      stop_(start_),
      total_duration_(0),
      last_duration_(0),
      min_duration_(INT32_MAX),
      max_duration_(0) {}

void Stopwatch::Start() { start_ = std::chrono::steady_clock::now(); }

void Stopwatch::Stop() {
  stop_ = std::chrono::steady_clock::now();
  last_duration_ = std::chrono::duration_cast<Micro>(stop_ - start_);
  total_duration_ += last_duration_;
  min_duration_ = std::min(min_duration_, last_duration_);
  max_duration_ = std::max(max_duration_, last_duration_);
  timing_count_++;
}

void Stopwatch::Reset() {
  start_ = std::chrono::steady_clock::now();
  stop_ = start_;
  last_duration_ = Micro(0);
  total_duration_ = Micro(0);
  min_duration_ = Micro(INT32_MAX);
  max_duration_ = Micro(0);
  timing_count_ = 0;
}

float Stopwatch::Duration() { return total_duration_.count() / 1000.0; }

float Stopwatch::LastDuration() { return last_duration_.count() / 1000.0; }

float Stopwatch::Min() { return min_duration_.count() / 1000.0; }

float Stopwatch::Max() { return max_duration_.count() / 1000.0; }

int Stopwatch::TimingCount() { return timing_count_; }

float Stopwatch::Fps() {
  return (float)(timing_count_) / (Duration() / 1000.0);
}

float Stopwatch::Average() {
  if (timing_count_ < 3) {
    return Duration() / timing_count_;
  } else {
    return (Duration() - Min() - Max()) / (timing_count_ - 2);
  }
}

std::ostream& operator<<(std::ostream& os, Stopwatch& stop_watch) {
  os << "count:" << stop_watch.TimingCount()
     << ", duration:" << std::setprecision(6) << stop_watch.Duration() << "ms"
     << ", min:" << std::setprecision(6) << stop_watch.Min() << "ms"
     << ", max:" << std::setprecision(6) << stop_watch.Max() << "ms"
     << ", average:" << std::setprecision(6) << stop_watch.Average() << "ms"
     << ", fps:" << std::setprecision(6) << stop_watch.Fps() << "/s"
     << std::endl;
  return os;
}
