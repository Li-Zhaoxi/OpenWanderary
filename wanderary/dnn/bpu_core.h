#pragma once
#include <map>
#include <string>
#include <vector>

#include <dnn/hb_dnn.h>
#include <wanderary/utils/enum_traits.h>
#include <wanderary/utils/json_utils.h>

namespace wdr::dnn {

ENUM_NUMBERED_REGISTER(HBDNNQuantiType,                                    //
                       ((kNONE, hbDNNQuantiType::NONE, "quanti_none"))     //
                       ((kSHIFT, hbDNNQuantiType::SHIFT, "quanti_shift"))  //
                       ((kSCALE, hbDNNQuantiType::SCALE, "quanti_scale"))  //
)
ENUM_CONVERSION_REGISTER(HBDNNQuantiType, HBDNNQuantiType::kNONE, "quanti_none")

ENUM_NUMBERED_REGISTER(
    HBDNNTensorLayout,                                               //
    ((kNHWC, hbDNNTensorLayout::HB_DNN_LAYOUT_NHWC, "layout_nhwc"))  //
    ((kNCHW, hbDNNTensorLayout::HB_DNN_LAYOUT_NCHW, "layout_nchw"))  //
    ((kNONE, hbDNNTensorLayout::HB_DNN_LAYOUT_NONE, "layout_none"))  //
)
ENUM_CONVERSION_REGISTER(HBDNNTensorLayout, HBDNNTensorLayout::kNONE,
                         "layout_nhwc")

ENUM_NUMBERED_REGISTER(
    HBDNNDataType,                                                           //
    ((kNONE, -1, "datatype_none"))                                           //
    ((kIMG_TYPE_Y, hbDNNDataType::HB_DNN_IMG_TYPE_Y, "imgtype_y"))           //
    ((kIMG_TYPE_NV12, hbDNNDataType::HB_DNN_IMG_TYPE_NV12, "imgtype_nv12"))  //
    ((kIMG_TYPE_NV12_SEPARATE, hbDNNDataType::HB_DNN_IMG_TYPE_NV12_SEPARATE,
      "imgtype_nv12_separate"))  //
    ((kIMG_TYPE_YUV444, hbDNNDataType::HB_DNN_IMG_TYPE_YUV444,
      "imgtype_yuv444"))                                                  //
    ((kIMG_TYPE_RGB, hbDNNDataType::HB_DNN_IMG_TYPE_RGB, "imgtype_rgb"))  //
    ((kIMG_TYPE_BGR, hbDNNDataType::HB_DNN_IMG_TYPE_BGR, "imgtype_bgr"))  //
    ((kTENSOR_TYPE_S4, hbDNNDataType::HB_DNN_TENSOR_TYPE_S4,
      "tensor_type_s4"))  //
    ((kTENSOR_TYPE_U4, hbDNNDataType::HB_DNN_TENSOR_TYPE_U4,
      "tensor_type_u4"))  //
    ((kTENSOR_TYPE_S8, hbDNNDataType::HB_DNN_TENSOR_TYPE_S8,
      "tensor_type_s8"))  //
    ((kTENSOR_TYPE_U8, hbDNNDataType::HB_DNN_TENSOR_TYPE_U8,
      "tensor_type_u8"))  //
    ((kTENSOR_TYPE_F16, hbDNNDataType::HB_DNN_TENSOR_TYPE_F16,
      "tensor_type_f16"))  //
    ((kTENSOR_TYPE_S16, hbDNNDataType::HB_DNN_TENSOR_TYPE_S16,
      "tensor_type_s16"))  //
    ((kTENSOR_TYPE_U16, hbDNNDataType::HB_DNN_TENSOR_TYPE_U16,
      "tensor_type_u16"))  //
    ((kTENSOR_TYPE_F32, hbDNNDataType::HB_DNN_TENSOR_TYPE_F32,
      "tensor_type_f32"))  //
    ((kTENSOR_TYPE_S32, hbDNNDataType::HB_DNN_TENSOR_TYPE_S32,
      "tensor_type_s32"))  //
    ((kTENSOR_TYPE_U32, hbDNNDataType::HB_DNN_TENSOR_TYPE_U32,
      "tensor_type_u32"))  //
    ((kTENSOR_TYPE_F64, hbDNNDataType::HB_DNN_TENSOR_TYPE_F64,
      "tensor_type_f64"))  //
    ((kTENSOR_TYPE_S64, hbDNNDataType::HB_DNN_TENSOR_TYPE_S64,
      "tensor_type_s64"))  //
    ((kTENSOR_TYPE_U64, hbDNNDataType::HB_DNN_TENSOR_TYPE_U64,
      "tensor_type_u64"))  //
    ((kTENSOR_TYPE_MAX, hbDNNDataType::HB_DNN_TENSOR_TYPE_MAX,
      "tensor_type_max"))  //

)
ENUM_CONVERSION_REGISTER(HBDNNDataType, HBDNNDataType::kNONE, "layout_nhwc")

using Json = wdr::json;

Json dump(const hbDNNQuantiShift &dtype);
Json dump(const hbDNNTensorShape &dtype);
Json dump(const hbDNNQuantiScale &dtype);
Json dump(const hbDNNTensorProperties &dtype);

// 加载网络
void readNets(const std::vector<std::string> &modelpaths,
              hbPackedDNNHandle_t *pPackedNets,
              std::map<std::string, hbDNNHandle_t> *netsMap);

// 释放网络
void releaseNets(hbPackedDNNHandle_t *pPackedNets);

// 读取网络输入输出属性
void readNetProperties(const hbDNNHandle_t dnn_handle, bool fetch_input,
                       std::vector<hbDNNTensorProperties> *properties);

// 内存分配
void createTensors(const std::vector<hbDNNTensorProperties> &properties,
                   bool autopadding, std::vector<hbDNNTensor> *tensors);
void createTensors(const hbDNNHandle_t dnn_handle, bool fetch_input,
                   bool autopadding, std::vector<hbDNNTensor> *tensors);
void createTensors(const hbDNNTensorProperties &property,
                   hbDNNTensor *bputensor);

// 内存释放
void releaseTensors(std::vector<hbDNNTensor> *tensors);

// 内存刷新，若upload=true,则为CPU刷新到BPU上，否则为BPU刷新到CPU上
void flushBPU(bool upload, hbDNNTensor *dst);

// 网络推理，两种模式，vector自动保证内存连续，而指针的方式需要时候需要自己注意下内存连续问题。
void forwardBPU(const hbDNNHandle_t dnn_handle,
                const std::vector<hbDNNTensor> &inTensors,
                std::vector<hbDNNTensor> *outTensors, int waiting_time = 0);

}  // namespace wdr::dnn
