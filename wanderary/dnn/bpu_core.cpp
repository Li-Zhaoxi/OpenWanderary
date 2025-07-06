#include "wanderary/dnn/bpu_core.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "wanderary/utils/path.h"

namespace wdr::dnn {

Json dump(const hbDNNQuantiShift &dtype) {
  Json res;
  res["dtype"] = "hbDNNQuantiShift";
  res["shiftLen"] = dtype.shiftLen;
  res["shiftData"] =
      std::vector<int>(dtype.shiftData, dtype.shiftData + dtype.shiftLen);
  return res;
}

Json dump(const hbDNNTensorShape &dtype) {
  CHECK_LE(dtype.numDimensions, HB_DNN_TENSOR_MAX_DIMENSIONS);
  Json res;
  res["dtype"] = "hbDNNTensorShape";
  res["dim"] = dtype.numDimensions;
  res["dimensionSize"] = std::vector<int32_t>(
      dtype.dimensionSize, dtype.dimensionSize + dtype.numDimensions);
  return res;
}

Json dump(const hbDNNQuantiScale &dtype) {
  Json res;
  res["dtype"] = "hbDNNQuantiScale";
  res["scaleLen"] = dtype.scaleLen;
  res["scaleData"] =
      std::vector<float>(dtype.scaleData, dtype.scaleData + dtype.scaleLen);
  res["zeroPointLen"] = dtype.zeroPointLen;
  res["zeroPointData"] = std::vector<int8_t>(
      dtype.zeroPointData, dtype.zeroPointData + dtype.zeroPointLen);

  return res;
}

Json dump(const hbDNNTensorProperties &dtype) {
  Json res;
  res["dtype"] = "hbDNNTensorProperties";
  res["validShape"] = dump(dtype.validShape);
  res["alignedShape"] = dump(dtype.alignedShape);
  res["tensorLayout"] =
      HBDNNTensorLayout2str(int2HBDNNTensorLayout(dtype.tensorLayout));
  res["tensorType"] = HBDNNDataType2str(int2HBDNNDataType(dtype.tensorType));
  res["shift"] = dump(dtype.shift);
  res["scale"] = dump(dtype.scale);
  res["quantiType"] =
      HBDNNQuantiType2str(int2HBDNNQuantiType(dtype.quantiType));
  res["quantizeAxis"] = dtype.quantizeAxis;
  res["alignedByteSize"] = dtype.alignedByteSize;
  res["stride"] = std::vector<int32_t>(
      dtype.stride, dtype.stride + HB_DNN_TENSOR_MAX_DIMENSIONS);

  return res;
}

void readNets(const std::vector<std::string> &modelpaths,
              hbPackedDNNHandle_t *pPackedNets,
              std::map<std::string, hbDNNHandle_t> *netsMap) {
  CHECK(*pPackedNets == nullptr);

  // 从文件初始化初始化Bin文件
  std::vector<const char *> cpaths;
  for (const auto &path : modelpaths) cpaths.push_back(path.c_str());
  CHECK_EQ(hbDNNInitializeFromFiles(pPackedNets, cpaths.data(), cpaths.size()),
           0);

  // 获取所有模型的handles
  netsMap->clear();
  const char **model_name_list;
  int model_count = 0;
  CHECK_EQ(hbDNNGetModelNameList(&model_name_list, &model_count, *pPackedNets),
           0);

  for (int i = 0; i < model_count; i++) {
    hbDNNHandle_t tmp;
    const std::string modelname(model_name_list[i]);
    CHECK_EQ(hbDNNGetModelHandle(&tmp, *pPackedNets, modelname.c_str()), 0);
    netsMap->insert(std::make_pair(modelname, std::move(tmp)));
  }
}

void releaseNets(hbPackedDNNHandle_t *pPackedNets) {
  CHECK(*pPackedNets != nullptr);
  CHECK_EQ(hbDNNRelease(*pPackedNets), 0);
  *pPackedNets = nullptr;
}

void readNetProperties(const hbDNNHandle_t dnn_handle, bool fetch_input,
                       std::vector<hbDNNTensorProperties> *properties) {
  int tensornum = 0;

  if (fetch_input) {
    CHECK_EQ(hbDNNGetInputCount(&tensornum, dnn_handle), 0);
  } else {
    CHECK_EQ(hbDNNGetOutputCount(&tensornum, dnn_handle), 0);
  }

  properties->resize(tensornum);

  for (int i = 0; i < tensornum; i++) {
    auto &property = properties->at(i);
    if (fetch_input) {
      CHECK_EQ(hbDNNGetInputTensorProperties(&property, dnn_handle, i), 0);
    } else {
      CHECK_EQ(hbDNNGetOutputTensorProperties(&property, dnn_handle, i), 0);
    }
  }
}

std::string getTensorName(const hbDNNHandle_t dnn_handle, int index,
                          bool input) {
  const char *input_name;
  if (input) {
    CHECK_EQ(hbDNNGetInputName(&input_name, dnn_handle, index), 0);
  } else {
    CHECK_EQ(hbDNNGetOutputName(&input_name, dnn_handle, index), 0);
  }
  return std::string(input_name);
}

void createTensors(const hbDNNHandle_t dnn_handle, bool fetch_input,
                   std::vector<hbDNNTensor> *tensors) {
  std::vector<hbDNNTensorProperties> properties;
  readNetProperties(dnn_handle, fetch_input, &properties);
  const int tensornum = properties.size();
  tensors->resize(tensornum);

  for (int i = 0; i < tensornum; i++) {
    if (fetch_input) {
      createInputTensors(properties[i], &tensors->at(i));
      LOG(INFO) << "Create input tensor: "
                << getTensorName(dnn_handle, i, /*input = */ true);
    } else {
      createOutputTensors(properties[i], &tensors->at(i));
      LOG(INFO) << "Create output tensor: "
                << getTensorName(dnn_handle, i, /*input = */ false);
    }
  }
}

void createInputTensors(const hbDNNTensorProperties &property,
                        hbDNNTensor *bputensor) {
  const int batch = property.alignedShape.dimensionSize[0];
  const int batch_size = property.alignedByteSize / batch;
  const auto tensor_type = property.tensorType;
  bputensor->properties = property;
  auto &tensor_properties = bputensor->properties;

  if (tensor_type == HB_DNN_IMG_TYPE_NV12 ||
      tensor_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    tensor_properties.alignedShape = property.validShape;
    if (tensor_type == HB_DNN_IMG_TYPE_NV12) {
      /* 分配NV12格式的内存
        这里失败也没关系，报错后会自动释放。
        官方分配失败并不会直接退出，因此需要主动释放模型内存
      */
      CHECK_EQ(hbSysAllocCachedMem(&bputensor->sysMem[0], batch_size * batch),
               0);
    } else if (tensor_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
      CHECK_EQ(hbSysAllocCachedMem(&bputensor->sysMem[0],
                                   batch_size * 2 / 3 * batch),
               0);
      CHECK_EQ(
          hbSysAllocCachedMem(&bputensor->sysMem[1], batch_size / 3 * batch),
          0);
    } else {
      LOG(FATAL) << "Unsupported tensor type: " << tensor_type;
    }
  } else {
    const int input_memsize = property.alignedByteSize;
    CHECK_EQ(hbSysAllocCachedMem(&bputensor->sysMem[0], input_memsize), 0);
    tensor_properties.alignedShape = property.validShape;
  }
}

void createOutputTensors(const hbDNNTensorProperties &property,
                         hbDNNTensor *bputensor) {
  bputensor->properties = property;
  const int output_memsize = property.alignedByteSize;
  CHECK_EQ(hbSysAllocCachedMem(&bputensor->sysMem[0], output_memsize), 0);
}

void bpuMemcpy(const cv::Mat &src, hbDNNTensor *dst) {
  CHECK(src.isContinuous());
  const auto &prop = dst->properties;
  const int size = prop.alignedByteSize;
  const int src_size = src.total() * src.elemSize();
  CHECK_LE(src_size, size);
  const auto tensor_type = prop.tensorType;

  switch (tensor_type) {
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
      memcpy(dst->sysMem[0].virAddr, src.data, src_size / 3 * 2);
      memcpy(dst->sysMem[1].virAddr, src.data + src_size / 3 * 2, src_size / 3);
      hbSysFlushMem(&dst->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
      hbSysFlushMem(&dst->sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
      break;
    case HB_DNN_IMG_TYPE_Y:
    case HB_DNN_IMG_TYPE_NV12:
    case HB_DNN_IMG_TYPE_YUV444:
    case HB_DNN_IMG_TYPE_RGB:
    case HB_DNN_IMG_TYPE_BGR:
    case HB_DNN_TENSOR_TYPE_U8:
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64:
    case HB_DNN_TENSOR_TYPE_U64:
      memcpy(dst->sysMem[0].virAddr, src.data, src_size);
      hbSysFlushMem(&dst->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
      break;
    default:
      LOG(FATAL) << "Unsupported tensor type: "
                 << HBDNNDataType2str(int2HBDNNDataType(tensor_type));
      break;
  }
}

void bpuMemcpy(hbDNNTensor *src, cv::Mat *dst) {
  hbSysFlushMem(&src->sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
  // 获取输出矩阵的维度信息
  constexpr int kTensorDimNum = 4;
  const auto &prop = src->properties;
  CHECK_EQ(prop.validShape.numDimensions, kTensorDimNum);
  int total_size = 1;
  int dims[kTensorDimNum];
  for (int i = 0; i < kTensorDimNum; i++) {
    dims[i] = prop.validShape.dimensionSize[i];
    total_size *= dims[i];
  }

  const auto data_type = prop.tensorType;
  // 创建输出矩阵并拷贝数据
  switch (data_type) {
    case HB_DNN_TENSOR_TYPE_S8:
      dst->create(kTensorDimNum, dims, CV_8S);
      memcpy(dst->data, src->sysMem[0].virAddr, total_size * sizeof(int8_t));
      break;
    case HB_DNN_TENSOR_TYPE_S16:
      dst->create(kTensorDimNum, dims, CV_16S);
      memcpy(dst->data, src->sysMem[0].virAddr, total_size * sizeof(int16_t));
      break;
    case HB_DNN_TENSOR_TYPE_S32:
      dst->create(kTensorDimNum, dims, CV_32S);
      memcpy(dst->data, src->sysMem[0].virAddr, total_size * sizeof(int32_t));
      break;
    case HB_DNN_TENSOR_TYPE_F32:
      dst->create(kTensorDimNum, dims, CV_32F);
      memcpy(dst->data, src->sysMem[0].virAddr, total_size * sizeof(float));
      break;
    default:
      LOG(FATAL) << "Unsupported tensor type: "
                 << HBDNNDataType2str(int2HBDNNDataType(data_type));
      break;
  }
}

void releaseTensors(bool input, std::vector<hbDNNTensor> *tensors) {
  for (auto &tensor : *tensors) {
    if (tensor.sysMem != nullptr) {
      CHECK_EQ(hbSysFreeMem(tensor.sysMem), 0);
      if (input &&
          tensor.properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
        CHECK_EQ(hbSysFreeMem(&(tensor.sysMem[1])), 0);
      }
    }
  }
}

void flushBPU(bool upload, hbDNNTensor *dst) {
  if (upload) {
    CHECK_EQ(hbSysFlushMem(&dst->sysMem[0], HB_SYS_MEM_CACHE_CLEAN), 0);
  } else {
    CHECK_EQ(hbSysFlushMem(&dst->sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE), 0);
  }
}

void forwardBPU(const hbDNNHandle_t dnn_handle,
                const std::vector<hbDNNTensor> &inTensors,
                std::vector<hbDNNTensor> *outTensors, int waiting_time) {
  hbDNNInferCtrlParam infer_ctrl_param;
  hbDNNTaskHandle_t task_handle = nullptr;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

  hbDNNTensor *_outTensors = outTensors->data();
  CHECK_EQ(hbDNNInfer(&task_handle, &_outTensors, inTensors.data(), dnn_handle,
                      &infer_ctrl_param),
           0);

  CHECK_EQ(hbDNNWaitTaskDone(task_handle, waiting_time), 0);
  CHECK_EQ(hbDNNReleaseTask(task_handle), 0);
}

}  // namespace wdr::dnn
