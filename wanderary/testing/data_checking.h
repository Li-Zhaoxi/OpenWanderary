#pragma once
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace wdr::testing {

template <typename DType>
void Check(const std::vector<DType> &pred, const std::vector<DType> &gt,
           double eps, const std::string &msg = "");

}  // namespace wdr::testing

#include <wanderary/testing/impl/data_checking.hpp>
