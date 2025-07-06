#pragma once
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <wanderary/structs/box.h>

#include <opencv2/opencv.hpp>

namespace wdr::testing {

template <typename DType>
void Check(const std::vector<DType> &pred, const std::vector<DType> &gt,
           double eps, const std::string &msg = "");

// 不支持自定义维度的比较方式
void CheckGeneralMat(const cv::Mat &pred, const cv::Mat &gt, double eps,
                     const std::string &msg = "");

void CheckNonGeneralMat(const cv::Mat &pred, const cv::Mat &gt, double eps,
                        const std::string &msg = "");

void Check(const std::set<std::string> &pred, const std::set<std::string> &gt,
           const std::string &msg = "");

void Check(const wdr::Box2D &pred, const wdr::Box2D &gt, double eps,
           const std::string &msg = "");

void UnorderedCheck(const std::vector<wdr::Box2D> &pred,
                    const std::vector<wdr::Box2D> &gt, double eps,
                    const std::string &msg = "");

void Check(const wdr::json &pred, const wdr::json &gt, double eps,
           const std::string &msg = "");

}  // namespace wdr::testing

#include <wanderary/testing/impl/data_checking.hpp>
