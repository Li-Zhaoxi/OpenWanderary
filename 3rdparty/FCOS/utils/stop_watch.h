// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

// Timing utils, record intervals between `Start` and `Stop`, and
// calculate min, max, average interval and fps and so on.

#ifndef _UTILS_STOP_WATCH_H_
#define _UTILS_STOP_WATCH_H_

#include <chrono>
#include <fstream>
#include <ostream>
#include <string>

typedef std::chrono::steady_clock::time_point Time;
typedef std::chrono::duration<u_int64_t, std::micro> Micro;

class Stopwatch {
 public:
  Stopwatch();

  /**
   * Current timestamp (microseconds)z
   * @return current timestamp
   */
  static u_int64_t CurrentTs();

  /**
   * Start timing
   */
  void Start();

  /**
   * Stop current timing
   */
  void Stop();

  /**
   * Reset timing,  clear all records (min, max, last_duration, total_duration)
   */
  void Reset();

  /**
   * Calculate fps as count/total_duration
   * @return fps
   */
  float Fps();

  /**
   * Caclucate average duration (total_duration-min-max)/(count-2)
   * @return average duration
   */
  float Average();

  /**
   * Total duration for all intervals
   * @return total duration
   */
  float Duration();

  /**
   * Min timing
   * @return min timing duration
   */
  float Min();

  /**
   * Max timing
   * @return max timing duration
   */
  float Max();

  /**
   * Last timing duration
   * @return last timing duration
   */
  float LastDuration();

  /**
   * Timing count
   * @return timing count
   */
  int TimingCount();

 private:
  Time start_;
  Time stop_;
  Micro total_duration_;
  Micro last_duration_;
  Micro min_duration_;
  Micro max_duration_;
  int timing_count_;
};

std::ostream& operator<<(std::ostream&, Stopwatch&);

#endif  // _UTILS_STOP_WATCH_H_
