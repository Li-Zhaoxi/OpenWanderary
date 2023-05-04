// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/ptq_fcos_post_process_method.h"

#include <arm_neon.h>

#include <iostream>
#include <queue>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/algorithm.h"
#include "utils/nms.h"

DEFINE_AND_REGISTER_METHOD(PTQFcosPostProcessMethod);

PTQFcosConfig default_ptq_fcos_config = {
    {{4, 8, 16, 32, 64}},
    80,
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"},
    ""};

struct ScoreId {
  float score;
  int id;
};

int PTQFcosPostProcessMethod::InitFromJsonString(const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "PTQFcosPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("topk")) {
    topk_ = document["topk"].GetInt();
  }

  if (document.HasMember("community_qat")) {
    community_qat_ = true;
  }

  if (document.HasMember("strides")) {
    rapidjson::Value &stride_value = document["strides"];
    auto strides_array = stride_value.GetArray();
    for (int i = 0; i < strides_array.Size(); ++i) {
      fcos_config_.strides[i] = strides_array[i].GetInt();
    }
  }

  if (document.HasMember("class_num")) {
    fcos_config_.class_num = document["class_num"].GetInt();
  }

  if (document.HasMember("det_name_list")) {
    // det_name_list
    fcos_config_.det_name_list = document["det_name_list"].GetString();
    ifs_.open(fcos_config_.det_name_list, std::ios::in);
    if (!ifs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << fcos_config_.det_name_list
                           << " failed!";
      return -1;
    }
    std::string line;
    fcos_config_.class_names.clear();
    while (std::getline(ifs_, line)) {
      fcos_config_.class_names.push_back(line);
    }
  }

  VLOG(EXAMPLE_DEBUG) << " topk: " << topk_
                      << ", class_num: " << fcos_config_.class_num
                      << ", score_threshold: " << score_threshold_
                      << ", stride {" << fcos_config_.strides[0] << ", "
                      << fcos_config_.strides[1] << ", "
                      << fcos_config_.strides[2] << ", "
                      << fcos_config_.strides[3] << ", "
                      << fcos_config_.strides[4] << "}";
  return 0;
}

PerceptionPtr PTQFcosPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  VLOG(EXAMPLE_DETAIL)
      << "PTQFcosPostProcessMethod DoProcess finished, predict result: "
      << *(perception.get());
  return perception;
}

void PTQFcosPostProcessMethod::GetBboxAndScoresScaleNHWC(
    std::vector<hbDNNTensor> &tensors,
    ImageTensor *image_tensor,
    std::vector<Detection> &dets) {
  int ori_h = image_tensor->ori_image_height;
  int ori_w = image_tensor->ori_image_width;
  int input_h = image_tensor->height();
  int input_w = image_tensor->width();
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  if (image_tensor->is_pad_resize) {
    // float scale_x = ori_h/input_h;
    // float scale_y = ori_w/input_w;
    // float scale_m  = std::max(scale_x, scale_y);
    // w_scale = scale_m;
    // h_scale = scale_m;
    float scale = ori_h > ori_w ? ori_h : ori_w;
    w_scale = scale / input_w;
    h_scale = scale / input_h;
  } else {
    w_scale = static_cast<float>(ori_w) / input_w;
    h_scale = static_cast<float>(ori_h) / input_h;
  }

  // fcos stride is {8, 16, 32, 64, 128}
  for (int i = 0; i < 5; i++) {
    auto *cls_data = reinterpret_cast<int32_t *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<int32_t *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<int32_t *>(tensors[i + 10].sysMem[0].virAddr);
    float *cls_scale = tensors[i].properties.scale.scaleData;
    float *bbox_scale = tensors[i + 5].properties.scale.scaleData;
    float *ce_scale = tensors[i + 10].properties.scale.scaleData;

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];
    int32_t bbox_c_stride{
        tensors[i + 5].properties.alignedShape.dimensionSize[3]};
    int32_t ce_c_stride{
        tensors[i + 10].properties.alignedShape.dimensionSize[3]};

    for (int h = 0; h < tensor_h; h++) {
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = (h * tensor_w + w) * ce_c_stride;
        float ce_data_offset =
            1.0 / (1.0 + exp(-ce_data[ce_offset] * ce_scale[0]));

        int cls_offset = (h * tensor_w + w) * tensor_c;
        ScoreId tmp_score = {cls_data[cls_offset] * cls_scale[0], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++) {
          int cls_index = cls_offset + cls_c;
          float score = cls_data[cls_index] * cls_scale[cls_c];
          if (score > tmp_score.score) {
            tmp_score.id = cls_c;
            tmp_score.score = score;
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data_offset);
        if (tmp_score.score <= score_threshold_) continue;

        // get detection box
        Detection detection;
        int index = bbox_c_stride * (h * tensor_w + w);
        auto &strides = fcos_config_.strides;

        detection.bbox.xmin =
            ((w + 0.5) * strides[i] - (bbox_data[index] * bbox_scale[0])) *
            w_scale;
        detection.bbox.ymin =
            ((h + 0.5) * strides[i] - (bbox_data[index + 1] * bbox_scale[1])) *
            h_scale;
        detection.bbox.xmax =
            ((w + 0.5) * strides[i] + (bbox_data[index + 2] * bbox_scale[2])) *
            w_scale;
        detection.bbox.ymax =
            ((h + 0.5) * strides[i] + (bbox_data[index + 3] * bbox_scale[3])) *
            h_scale;

        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

void PTQFcosPostProcessMethod::GetBboxAndScoresNoneNHWC(
    std::vector<hbDNNTensor> &tensors,
    ImageTensor *image_tensor,
    std::vector<Detection> &dets) {
  int ori_h = image_tensor->ori_image_height;
  int ori_w = image_tensor->ori_image_width;
  int input_h = image_tensor->height();
  int input_w = image_tensor->width();
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  if (image_tensor->is_pad_resize) {
    float scale = ori_h > ori_w ? ori_h : ori_w;
    w_scale = scale / input_w;
    h_scale = scale / input_h;
  } else {
    w_scale = static_cast<float>(ori_w) / input_w;
    h_scale = static_cast<float>(ori_h) / input_h;
  }

  // fcos stride is {8, 16, 32, 64, 128}
  for (int i = 0; i < 5; i++) {
    auto *cls_data = reinterpret_cast<float *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10].sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];

    for (int h = 0; h < tensor_h; h++) {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        int cls_offset = ce_offset * tensor_c;
        ScoreId tmp_score = {cls_data[cls_offset], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++) {
          int cls_index = cls_offset + cls_c;
          if (cls_data[cls_index] > tmp_score.score) {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_threshold_) continue;

        // get detection box
        Detection detection;
        int index = 4 * (h * tensor_w + w);
        auto &strides = fcos_config_.strides;

        detection.bbox.xmin =
            ((w + 0.5) * strides[i] - bbox_data[index]) * w_scale;
        detection.bbox.ymin =
            ((h + 0.5) * strides[i] - bbox_data[index + 1]) * h_scale;
        detection.bbox.xmax =
            ((w + 0.5) * strides[i] + bbox_data[index + 2]) * w_scale;
        detection.bbox.ymax =
            ((h + 0.5) * strides[i] + bbox_data[index + 3]) * h_scale;

        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

void PTQFcosPostProcessMethod::GetBboxAndScoresNoneNCHW(
    std::vector<hbDNNTensor> &tensors,
    ImageTensor *image_tensor,
    std::vector<Detection> &dets) {
  int ori_h = image_tensor->ori_image_height;
  int ori_w = image_tensor->ori_image_width;
  int input_h = image_tensor->height();
  int input_w = image_tensor->width();
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  if (image_tensor->is_pad_resize) {
    // float scale = ori_h > ori_w ? ori_h : ori_w;
    // w_scale = scale / input_w;
    // h_scale = scale / input_h;
    float scale_1 = ori_h*1.0 / input_h;
    float scale_2 = ori_w*1.0 / input_w;
    float scale_m = std::max(scale_1, scale_2);
    w_scale = scale_m;
    h_scale = scale_m;
  } else {
    w_scale = static_cast<float>(ori_w) / input_w;
    h_scale = static_cast<float>(ori_h) / input_h;
  }
  std::cout<<"ori_h: "<<ori_h<<" , ori_w: "<<ori_w<<std::endl;
  std::cout<<"input_h: "<<input_h<<" , input_w: "<<input_w<<std::endl;
  std::cout<<"w_scale: "<<w_scale<<" , h_scale: "<<h_scale<<std::endl;

  for (int i = 0; i < 5; i++) {
    auto *cls_data = reinterpret_cast<float *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10].sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_c = shape[1];
    int tensor_h = shape[2];
    int tensor_w = shape[3];
    int aligned_hw = tensor_h * tensor_w;

    for (int h = 0; h < tensor_h; h++) {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        ScoreId tmp_score = {cls_data[offset + w], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++) {
          int cls_index = cls_c * aligned_hw + offset + w;
          if (cls_data[cls_index] > tmp_score.score) {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_threshold_) continue;

        // get detection box
        auto &strides = fcos_config_.strides;
        Detection detection;
        detection.bbox.xmin =
            ((w + 0.5) * strides[i] - bbox_data[offset + w]) * w_scale;
        detection.bbox.ymin =
            ((h + 0.5) * strides[i] - bbox_data[1 * aligned_hw + offset + w]) *
            h_scale;
        detection.bbox.xmax =
            ((w + 0.5) * strides[i] + bbox_data[2 * aligned_hw + offset + w]) *
            w_scale;
        detection.bbox.ymax =
            ((h + 0.5) * strides[i] + bbox_data[3 * aligned_hw + offset + w]) *
            h_scale;

        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

static inline uint32x4x4_t CalculateIndex(uint32_t idx,
                                          float32x4_t a,
                                          float32x4_t b,
                                          uint32x4x4_t c) {
  uint32x4_t mask{0};
  mask = vcltq_f32(b, a);
  uint32x4_t vec_idx = {idx, idx + 1, idx + 2, idx + 3};
  uint32x4x4_t res = {{vbslq_u32(mask, vec_idx, c.val[0]), 0, 0, 0}};
  return res;
}

static inline float32x2_t CalculateMax(float32x4_t in) {
  auto pmax = vpmax_f32(vget_high_f32(in), vget_low_f32(in));
  return vpmax_f32(pmax, pmax);
}

static inline uint32_t CalculateVectorIndex(uint32x4x4_t vec_res_idx,
                                            float32x4_t vec_res_value) {
  uint32x4_t res_idx_mask{0};
  uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);

  auto pmax = CalculateMax(vec_res_value);
  auto mask = vceqq_f32(vec_res_value, vcombine_f32(pmax, pmax));
  res_idx_mask = vandq_u32(vec_res_idx.val[0], mask);
  res_idx_mask = vaddq_u32(res_idx_mask, mask_ones);
  auto pmin =
      vpmin_u32(vget_high_u32(res_idx_mask), vget_low_u32(res_idx_mask));
  pmin = vpmin_u32(pmin, pmin);
  uint32_t res = vget_lane_u32(pmin, 0);
  return (res - 0xFFFFFFFF);
}

static std::pair<float, int> MaxScoreID(int32_t *input,
                                        float *scale,
                                        int length) {
  float init_res_value = input[0] * scale[0];
  float32x4_t vec_res_value = vdupq_n_f32(init_res_value);
  uint32x4x4_t vec_res_idx{{0}};
  int i = 0;
  for (; i <= (length - 4); i += 4) {
    int32x4_t vec_input = vld1q_s32(input + i);
    float32x4_t vec_scale = vld1q_f32(scale + i);

    float32x4_t vec_elements = vmulq_f32(vcvtq_f32_s32(vec_input), vec_scale);
    float32x4_t temp_vec_res_value = vmaxq_f32(vec_elements, vec_res_value);
    vec_res_idx =
        CalculateIndex(i, temp_vec_res_value, vec_res_value, vec_res_idx);
    vec_res_value = temp_vec_res_value;
  }

  uint32_t idx = CalculateVectorIndex(vec_res_idx, vec_res_value);
  float res = vget_lane_f32(CalculateMax(vec_res_value), 0);

  // Compute left elements
  for (; i < length; ++i) {
    float score = input[i] * scale[i];
    if (score > res) {
      idx = i;
      res = score;
    }
  }
  std::pair<float, int> result_id_score = {res, idx};
  return result_id_score;
}

void PTQFcosPostProcessMethod::CqatGetBboxAndScoresScaleNHWC(
    std::vector<hbDNNTensor> &tensors,
    ImageTensor *image_tensor,
    std::vector<Detection> &dets) {
  int ori_h = image_tensor->ori_image_height;
  int ori_w = image_tensor->ori_image_width;
  int input_h = image_tensor->height();
  int input_w = image_tensor->width();
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  if (image_tensor->is_pad_resize) {
    float scale = ori_h > ori_w ? ori_h : ori_w;
    w_scale = scale / input_w;
    h_scale = scale / input_h;
  } else {
    w_scale = static_cast<float>(ori_w) / input_w;
    h_scale = static_cast<float>(ori_h) / input_h;
  }

  // fcos stride is {8, 16, 32, 64, 128}
  for (int i = 0; i < 5; i++) {
    auto *cls_data = reinterpret_cast<int32_t *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<int32_t *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<int32_t *>(tensors[i + 10].sysMem[0].virAddr);
    float *cls_scale = tensors[i].properties.scale.scaleData;
    float *bbox_scale = tensors[i + 5].properties.scale.scaleData;
    float *ce_scale = tensors[i + 10].properties.scale.scaleData;

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];
    int32_t bbox_c_stride{
        tensors[i + 5].properties.alignedShape.dimensionSize[3]};
    int32_t ce_c_stride{
        tensors[i + 10].properties.alignedShape.dimensionSize[3]};

    for (int h = 0; h < tensor_h; h++) {
      for (int w = 0; w < tensor_w; w++) {
        // get score
        int ce_offset = (h * tensor_w + w) * ce_c_stride;
        float ce_data_offset =
            1.0 / (1.0 + exp(-ce_data[ce_offset] * ce_scale[0]));
        // argmax + neon
        int cls_offset = (h * tensor_w + w) * tensor_c;
        auto max_score_id =
            MaxScoreID(cls_data + cls_offset, cls_scale, tensor_c);

        // filter
        float cls_data_offset = 1.0 / (1.0 + exp(-max_score_id.first));
        float score = std::sqrt(cls_data_offset * ce_data_offset);
        if (score <= score_threshold_) continue;

        // get detection box
        Detection detection;
        int index = bbox_c_stride * (h * tensor_w + w);
        auto &strides = fcos_config_.strides;

        float xmin = std::max(0.f, bbox_data[index] * bbox_scale[0]);
        float ymin = std::max(0.f, bbox_data[index + 1] * bbox_scale[1]);
        float xmax = std::max(0.f, bbox_data[index + 2] * bbox_scale[2]);
        float ymax = std::max(0.f, bbox_data[index + 3] * bbox_scale[3]);

        detection.bbox.xmin = ((w + 0.5) - xmin) * strides[i] * w_scale;
        detection.bbox.ymin = ((h + 0.5) - ymin) * strides[i] * h_scale;
        detection.bbox.xmax = ((w + 0.5) + xmax) * strides[i] * w_scale;
        detection.bbox.ymax = ((h + 0.5) + ymax) * strides[i] * h_scale;

        detection.score = score;
        detection.id = max_score_id.second;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

int PTQFcosPostProcessMethod::PostProcess(std::vector<hbDNNTensor> &tensors,
                                          ImageTensor *image_tensor,
                                          Perception *perception) {
  perception->type = Perception::DET;
  // TODO(daofu.zhang) check namelist for model output, compare with config file
  int h_index, w_index, c_index;
  int ret = get_tensor_hwc_index(&tensors[0], &h_index, &w_index, &c_index);
  if (ret != 0 &&
      fcos_config_.class_names.size() !=
          tensors[0].properties.alignedShape.dimensionSize[c_index]) {
    VLOG(EXAMPLE_SYSTEM)
        << "User det_name_list in config file: '" << fcos_config_.det_name_list
        << "', is not compatible with this model. "
        << fcos_config_.class_names.size()
        << " != " << tensors[0].properties.alignedShape.dimensionSize[c_index]
        << ".";
  }
  for (int i = 0; i < tensors.size(); i++) {
    hbSysFlushMem(&(tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  std::vector<std::vector<ScoreId>> scores;
  std::vector<Detection> dets;
  auto quanti_type = tensors[0].properties.quantiType;

  if (community_qat_) {
    CqatGetBboxAndScoresScaleNHWC(tensors, image_tensor, dets);
  } else {
    if (quanti_type == hbDNNQuantiType::SCALE) {
      if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        GetBboxAndScoresScaleNHWC(tensors, image_tensor, dets);
      } else if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        VLOG(EXAMPLE_SYSTEM) << "NCHW type is not supported.";
      } else {
        VLOG(EXAMPLE_SYSTEM) << "tensor layout error.";
      }
    } else if (quanti_type == hbDNNQuantiType::NONE) {
      if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        GetBboxAndScoresNoneNHWC(tensors, image_tensor, dets);
      } else if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        GetBboxAndScoresNoneNCHW(tensors, image_tensor, dets);
      } else {
        VLOG(EXAMPLE_SYSTEM) << "tensor layout error.";
      }
    } else {
      VLOG(EXAMPLE_SYSTEM) << "error quanti_type: " << quanti_type;
      return -1;
    }
  }
  yolo5_nms(dets, 0.3, topk_, perception->det, false);
  return 0;
}
