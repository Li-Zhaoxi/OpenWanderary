#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/utils/common_utils.h"

TEST(CRC, ExtendAbslCRC32c) {
  const uint64_t length = 4416964;
  const uint32_t gt_crc = 3275706847;
  const uint32_t mask_delta = 0xa282ead8ul;

  const uint32_t crc = wdr::ExtendAbslCRC32c(
      0, reinterpret_cast<const char*>(&length), sizeof(length));
  const uint32_t masked_crc = ((crc >> 15) | (crc << 17)) + mask_delta;

  EXPECT_EQ(masked_crc, gt_crc);
}
