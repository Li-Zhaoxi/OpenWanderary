#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/process/process_base.h"
#include "wanderary/process/processors/format_image.h"
#include "wanderary/testing/data_checking.h"
#include "wanderary/testing/data_convertor.h"
#include "wanderary/utils/file_io.h"
#include "wanderary/utils/time_manager.h"

using FormatImage = wdr::proc::FormatImage;
using ProcessRecorder = wdr::proc::ProcessRecorder;
using TimeManager = wdr::TimerManager;

TEST(ProcessBase, TestFormatImage) {
  const std::string cfg = R"({
    "type": "letter_box",
    "width": 640,
    "height": 640,
    "cvt_nv12": true
  })";

  TimeManager time_manaer;

  const std::string imgpath = "../../test_data/media/zidane.jpg";
  const std::string gtpath = "../../test_data/process/zidane_fmtimg_nv12.bin";

  cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);
  const auto fmtgt = wdr::ReadBytesFromFile<uchar>(gtpath);

  FormatImage proc(wdr::utils::json::parse(cfg));
  ProcessRecorder recorder;
  time_manaer.start("FormatImage");
  proc.Forward(&img, &recorder);
  time_manaer.stop("FormatImage");
  time_manaer.printStatistics();

  ASSERT_TRUE(recorder.affine.has_value());
  const auto &parms = recorder.affine.value();

  EXPECT_NEAR(parms.x_scale, 0.5, 1e-6);
  EXPECT_NEAR(parms.y_scale, 0.5, 1e-6);
  EXPECT_EQ(parms.x_shift, 0);
  EXPECT_EQ(parms.y_shift, 140);

  wdr::testing::Check<uchar>(wdr::testing::Convertor(img), fmtgt, 0);
}
