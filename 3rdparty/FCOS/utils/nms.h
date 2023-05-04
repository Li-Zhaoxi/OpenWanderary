// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_NMS_H_
#define _UTILS_NMS_H_

#include <vector>

#include "base/perception_common.h"

/**
 * Non-maximum suppression
 * @param[in] input
 * @param[in] iou_threshold
 * @param[in] top_k
 * @param[out] result
 * @param[in] suppress
 */
void nms(std::vector<Detection> &input,
         float iou_threshold,
         int top_k,
         std::vector<Detection> &result,
         bool suppress = false);

/**
 * Non-maximum suppression
 * @param[in] input
 * @param[in] iou_threshold
 * @param[in] top_k
 * @param[out] result
 * @param[in] suppress
 */
void yolo5_nms(std::vector<Detection> &input,
               float iou_threshold,
               int top_k,
               std::vector<Detection> &result,
               bool suppress);
#endif  // _UTILS_NMS_H_
