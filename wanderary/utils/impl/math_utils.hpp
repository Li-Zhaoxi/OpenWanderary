#pragma once
#include <algorithm>
#include <vector>

#include <wanderary/utils/math_utils.h>

namespace wdr {

template <typename DType>
void softmax(DType* data, int num) {
  const DType max_val = max(data, num);
  std::vector<DType> exp_data(num);

  DType* exp_data_ptr = exp_data.data();
  DType sum_shifted = 0;

  for (int i = 0; i < num; ++i) {
    exp_data_ptr[i] = exp(data[i] - max_val);
    sum_shifted += exp_data_ptr[i];
  }

  for (int i = 0; i < num; ++i) data[i] = exp_data_ptr[i] / sum_shifted;
}

template <typename DType>
DType weight_sum(const DType* data, const DType* w, int num) {
  DType sum_val = 0;
  for (int i = 0; i < num; ++i) sum_val += data[i] * w[i];
  return sum_val;
}

}  // namespace wdr
