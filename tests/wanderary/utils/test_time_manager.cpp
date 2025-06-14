#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/utils/time_manager.h"

using TimerManager = wdr::TimerManager;
using AutoScopeTimer = wdr::AutoScopeTimer;

TEST(TimeManager, RecordStages) {
  TimerManager time_manager;
  time_manager.start("stage1");
  sleep(1);
  time_manager.start("stage2");
  sleep(1);
  time_manager.stop("stage2");
  sleep(1);
  time_manager.stop("stage1");

  time_manager.printStatistics();

  EXPECT_NEAR(time_manager.getDuration("stage1"), 3000, 1);
  EXPECT_NEAR(time_manager.getDuration("stage2"), 1000, 1);
}

TEST(TimeManager, AutoScopeTimer) {
  TimerManager time_manager;

  {
    AutoScopeTimer timer("stage1", &time_manager);
    sleep(1);
    {
      AutoScopeTimer timer("stage2", &time_manager);
      sleep(1);
    }
  }

  time_manager.printStatistics();

  EXPECT_NEAR(time_manager.getDuration("stage1"), 2000, 1);
  EXPECT_NEAR(time_manager.getDuration("stage2"), 1000, 1);
}

void TestFun1() {
  AutoScopeTimer timer("stage1", &wdr::GlobalTimerManager());
  sleep(1);
}

void TestFun2() {
  AutoScopeTimer timer("stage2", &wdr::GlobalTimerManager());
  sleep(2);
}

TEST(TimeManager, GlobalTimer) {
  TestFun1();
  TestFun2();

  const auto &timer = wdr::GlobalTimerManager();
  timer.printStatistics();

  EXPECT_NEAR(timer.getDuration("stage1"), 1000, 1);
  EXPECT_NEAR(timer.getDuration("stage2"), 2000, 1);
}
