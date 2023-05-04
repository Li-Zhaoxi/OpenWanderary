// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _BASE_COMMON_DEF_H_
#define _BASE_COMMON_DEF_H_

#include <float.h>
#include <sys/time.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "glog/logging.h"

#define RED_COMMENT_START "\033[31m "
#define RED_COMMENT_END " \033[0m\n"
#define YELLOW_COMMENT_START "\033[33m"
#define YELLOW_COMMENT_END "\033[0m\n"

typedef std::chrono::steady_clock::time_point Time;
typedef std::chrono::duration<u_int64_t, std::micro> Micro;

enum VLOG_LEVEL {
  EXAMPLE_SYSTEM = 0,
  EXAMPLE_REPORT = 1,
  EXAMPLE_DETAIL = 2,
  EXAMPLE_DEBUG = 3
};

/**
 * Record the start time in the function
 */
#define MODULE_PERF_START(name)                             \
  static std::atomic<int> perf_count##name(0);              \
  int perf_local_cnt##name = perf_count##name.fetch_add(1); \
  struct timeval start_time##name;                          \
  gettimeofday(&start_time##name, NULL);

/**
 * Record the end time in the function
 * print end_time - start_time
 */
#define MODULE_PERF_END(name)                                                  \
  struct timeval end_time##name;                                               \
  gettimeofday(&end_time##name, NULL);                                         \
  float time_elapsed##name =                                                   \
      ((end_time##name.tv_sec - start_time##name.tv_sec) * 1000000 +           \
       (end_time##name.tv_usec - start_time##name.tv_usec)) /                  \
      1000.0;                                                                  \
  if (perf_local_cnt##name % 10 == 0) {                                        \
    VLOG(EXAMPLE_REPORT) << RED_COMMENT_START << "method name:" << #name       \
                         << " time:" << time_elapsed##name << RED_COMMENT_END; \
  }

/**
 * start_time = the first frame start time, in input_plugin
 * end_time = current time
 * count = total frame
 * fps = count / ( end_time - start_time)
 */
#define FPS_PERF_RECORD                                                    \
  const int statistic_cycle /*make it configurable*/ = 100;                \
  fps_count += 1;                                                          \
  if (fps_count % statistic_cycle == 0) {                                  \
    if (fps_count != 1 /* ignore first statistic cycle */) {               \
      Time fps_end = std::chrono::steady_clock::now();                     \
      double total_time_elapsed =                                          \
          std::chrono::duration_cast<Micro>(fps_end - fps_start).count() / \
          1000.0;                                                          \
      VLOG(EXAMPLE_REPORT) << RED_COMMENT_START << "frame_rate: "          \
                           << 1000 * statistic_cycle / total_time_elapsed  \
                           << RED_COMMENT_END;                             \
    }                                                                      \
    fps_start = std::chrono::steady_clock::now();                          \
  }

#endif  // _BASE_COMMON_DEF_H_
