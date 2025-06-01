#pragma once
#include <string>
#include <type_traits>
#include <vector>

#include <wanderary/testing/data_checking.h>

namespace wdr::testing {

template <typename DType>
void Check(const std::vector<DType> &pred, const std::vector<DType> &gt,
           double eps, const std::string &msg) {
  const int num = gt.size();
  EXPECT_EQ(num, pred.size());
  if (num != pred.size()) return;
  for (int i = 0; i < num; ++i) {
    if constexpr (std::is_floating_point_v<DType>) {
      EXPECT_NEAR(pred[i], gt[i], eps) << "(idx: " << i << ")->" << msg;
    } else {
      EXPECT_EQ(pred[i], gt[i]) << msg << "(idx: " << i << ")->" << msg;
    }
  }
}

}  // namespace wdr::testing
