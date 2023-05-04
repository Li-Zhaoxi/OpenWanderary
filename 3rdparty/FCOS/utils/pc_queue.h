// Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_PC_QUEUE_H_
#define _UTILS_PC_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class PCQueue {
 public:
  explicit PCQueue(int capacity = INT32_MAX) : capacity_(capacity) {}

  void put(T t) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      cv_.wait(lck, [this] { return q_.size() <= capacity_; });
      q_.push(t);
    }
    cv_.notify_all();
  }

  bool get(T& t, int timeout = 0) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      if (timeout > 0) {
        cv_.wait_for(lck, std::chrono::milliseconds(timeout), [this] {
          return !this->q_.empty();
        });
      } else {
        cv_.wait(lck, [this] { return !q_.empty(); });
      }
      if (q_.empty()) {
        return false;
      } else {
        t = q_.front();
        q_.pop();
      }
    }
    cv_.notify_all();
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> q_;
  int capacity_;
};

#endif  // _UTILS_PC_QUEUE_H_
