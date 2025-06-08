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
  const int pathnum = modelpaths.size();
  const char **cpaths = new const char *[pathnum];
  for (int k = 0; k < pathnum; k++) {
    CHECK(wdr::path::exist(modelpaths[k], true));
    cpaths[k] = modelpaths[k].c_str();
  }
  CHECK_EQ(hbDNNInitializeFromFiles(pPackedNets, cpaths, pathnum), 0);
  delete[] cpaths;

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

void createTensors(const std::vector<hbDNNTensorProperties> &properties,
                   bool autopadding, std::vector<hbDNNTensor> *tensors) {
  const int tensornum = properties.size();
  tensors->resize(tensornum);
  for (int i = 0; i < tensornum; i++) {
    auto &usage_tensor = tensors->at(i);
    usage_tensor.properties = properties[i];

    int memSize = usage_tensor.properties.alignedByteSize;
    CHECK_EQ(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), 0);

    if (autopadding)
      usage_tensor.properties.alignedShape = usage_tensor.properties.validShape;
  }
}

void createTensors(const hbDNNHandle_t dnn_handle, bool fetch_input,
                   bool autopadding, std::vector<hbDNNTensor> *tensors) {
  std::vector<hbDNNTensorProperties> properties;
  readNetProperties(dnn_handle, fetch_input, &properties);
  createTensors(properties, autopadding, tensors);
}

void createTensors(const hbDNNTensorProperties &property,
                   hbDNNTensor *bputensor) {
  bputensor->properties = property;
  int memSize = bputensor->properties.alignedByteSize;
  CHECK_EQ(hbSysAllocCachedMem(&bputensor->sysMem[0], memSize), 0);
}

void releaseTensors(std::vector<hbDNNTensor> *tensors) {
  for (auto &tensor : *tensors) {
    CHECK_EQ(hbSysFreeMem(&(tensor.sysMem[0])), 0);
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
