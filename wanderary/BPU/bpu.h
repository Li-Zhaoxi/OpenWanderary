#ifndef BPU_HPP_
#define BPU_HPP_


#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <unordered_map>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/opencv.hpp>

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  {                                                                  \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      LOG(ERROR) << errmsg << ", error code:" << ret_code;           \
      abort();                                                       \
    }                                                                \
  }


namespace wdr
{

namespace BPU
{
void readNets(const std::vector<std::string> &modelpaths, 
              hbPackedDNNHandle_t &pPackedNets,
              std::unordered_map<std::string, hbDNNHandle_t> &netsMap);

void readNetProperties(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensorProperties> &properties, bool input);

void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors);

void bpuMemcpy(hbDNNTensor &dst, const uint8_t *src, int memsize = -1);
void bpuMemcpy(uint8_t *dst, hbDNNTensor &src, int memsize = -1);

// cv::Mat转Tensor，用于转换输入的Mat到Tensor中
void bpuMemcpy(const cv::Mat &src, hbDNNTensor &dst);

// Tensor转cv:Mat，用于转换输出Tensor到Mat矩阵
void bpuMemcpy(hbDNNTensor &src, cv::Mat &dst);


void forward(const hbDNNHandle_t dnn_handle, const std::vector<hbDNNTensor> &inTensors, std::vector<hbDNNTensor> &outTensors, int waiting_time = 0);

void releaseTensors(std::vector<hbDNNTensor> &tensors);
void releaseNets(hbPackedDNNHandle_t &pPackedNets);



// BPU Tensor Info Show
void showhbDNNQuantiShift(const hbDNNQuantiShift &c1);
void showhbDNNTensorShape(const hbDNNTensorShape &c1);
void showhbDNNQuantiScale(const hbDNNQuantiScale &c1);
void showhbDNNQuantiType(const hbDNNQuantiType &c1);
void showhbDNNTensorLayout(const hbDNNTensorLayout &c1);
void showhbDNNDataType(const hbDNNDataType &c1);
void showhbDNNTensorProperties(const hbDNNTensorProperties &c1);


// Pre-processing functions

// 根据Tensor的属性，自动将输入的BGR图像，转为目标Tensor
// 目标Tensor属性仅支持BGR/RGB, HWC/CHW, NV12由于存在通道错误问题不增加相应的转换
void cvtImage2Tensor(const cv::Mat &img, hbDNNTensor &tensor);



// Post-processing functions


// 适用于二分类分割结果，后续有多分类需求，考虑优化这个函数或者改进这个函数
// 1x2xhxw->hxw
void parseBinarySegmentResult(const hbDNNTensor &tensor, cv::Mat &pred);


} // end BPU

// class BPUModels
// {
// public:
//   explicit BPUModels(const std::string &bin_path, const int model_num = 1);

// protected:

// private:
//   static void prepareInputOutputTensor(const hbDNNHandle_t &dnn_handle, std::vector<hbDNNTensor> &tensors, bool input);
  
// private:
//   hbPackedDNNHandle_t packed_dnn_handle;
// 	std::vector<hbDNNHandle_t> dnn_handles;
//   std::vector<std::vector<hbDNNTensor>> all_intensor, all_outtensor;
// };




} // end wdr


#endif