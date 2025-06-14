#pragma once

#include <algorithm>
namespace wdr {

template <typename DType>
DType max(const DType* data, int num) {
  return *std::max_element(data, data + num);
}

template <typename DType>
void softmax(DType* data, int num);

template <typename DType>
DType weight_sum(const DType* data, const DType* w, int num);

}  // namespace wdr

#include <wanderary/utils/impl/math_utils.hpp>
