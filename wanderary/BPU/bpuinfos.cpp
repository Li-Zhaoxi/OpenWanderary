#include <BPU/bpu.h>
#include <sstream>
#include <opencv2/opencv.hpp>

namespace wdr
{

namespace BPU
{

void showhbDNNQuantiShift(const hbDNNQuantiShift &c1)
{
  LOG(INFO) << "[hbDNNQuantiShift] shiftLen: " << c1.shiftLen 
            << ", shiftData: " 
            << cv::Mat(1, c1.shiftLen, CV_8UC1, c1.shiftData);
}

void showhbDNNTensorShape(const hbDNNTensorShape &c1)
{
  std::stringstream ss;
  ss << "[hbDNNTensorShape] dim: " << c1.numDimensions << " [";
  for(int k = 0; k < c1.numDimensions; k++)
  {
    ss << c1.dimensionSize[k];
    if (k < c1.numDimensions - 1)
      ss << "x";
  }
  ss << "]";
  LOG(INFO) << ss.str();
}

void showhbDNNQuantiScale(const hbDNNQuantiScale &c1)
{
  std::stringstream ss;
  ss << "[hbDNNQuantiScale] ";
  ss << "scaleLen: " << c1.scaleLen << ", scaleData: " << c1.scaleData << " ";
  if (c1.scaleLen != 0 && c1.scaleData)
    ss << cv::Mat(1, c1.scaleLen, CV_32FC1, c1.scaleData);
  else 
    ss << "[]";
  
  ss << ", zeroPointLen: " << c1.zeroPointLen << ", zeroPointData: " << " ";
  if (c1.zeroPointLen != 0 && c1.zeroPointData)
    ss << cv::Mat(1, c1.zeroPointLen, CV_8SC1, c1.zeroPointData);
  else 
    ss << "[]";

  LOG(INFO) << ss.str();
}

void showhbDNNQuantiType(const hbDNNQuantiType &c1)
{
  std::stringstream ss;
  ss << "[hbDNNQuantiType]: ";
  if (c1 == NONE)
    ss << "NONE";
  else if (c1 == SHIFT)
    ss << "SHIFT";
  else if (c1 == SCALE)
    ss << "SCALE";
  else 
    ss << c1;
  LOG(INFO) << ss.str();
}

void showhbDNNTensorLayout(const hbDNNTensorLayout &c1)
{
  std::stringstream ss;
  ss << "[hbDNNTensorLayout]: ";
  switch(c1)
  {
    case HB_DNN_LAYOUT_NHWC:
      ss << "HB_DNN_LAYOUT_NHWC";
      break;
    case HB_DNN_LAYOUT_NCHW:
      ss << "HB_DNN_LAYOUT_NCHW";
      break;
    case HB_DNN_LAYOUT_NONE:
      ss << "HB_DNN_LAYOUT_NONE";
      break;
    default:
      CV_Error(cv::Error::StsBadArg, "Wrong hbDNNTensorLayout value:" + std::to_string(c1));
      break;
  }

  LOG(INFO) << ss.str();
}

void showhbDNNDataType(const hbDNNDataType &c1)
{
  std::stringstream ss;
  ss << "[hbDNNDataType]: ";
  switch(c1)
  {
    case HB_DNN_IMG_TYPE_Y:
      ss << "HB_DNN_IMG_TYPE_Y";
      break;
    case HB_DNN_IMG_TYPE_NV12:
      ss << "HB_DNN_IMG_TYPE_NV12";
      break;
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
      ss << "HB_DNN_IMG_TYPE_NV12_SEPARATE";
      break;
    case HB_DNN_IMG_TYPE_YUV444:
      ss << "HB_DNN_IMG_TYPE_YUV444";
      break;
    case HB_DNN_IMG_TYPE_RGB:
      ss << "HB_DNN_IMG_TYPE_RGB";
      break;
    case HB_DNN_IMG_TYPE_BGR:
      ss << "HB_DNN_IMG_TYPE_BGR";
      break;
    case HB_DNN_TENSOR_TYPE_S4:
      ss << "HB_DNN_TENSOR_TYPE_S4";
      break;
    case HB_DNN_TENSOR_TYPE_U4:
      ss << "HB_DNN_TENSOR_TYPE_U4";
      break;
    case HB_DNN_TENSOR_TYPE_S8:
      ss << "HB_DNN_TENSOR_TYPE_S8";
      break;
    case HB_DNN_TENSOR_TYPE_U8:
      ss << "HB_DNN_TENSOR_TYPE_U8";
      break;
    case HB_DNN_TENSOR_TYPE_F16:
      ss << "HB_DNN_TENSOR_TYPE_F16";
      break;
    case HB_DNN_TENSOR_TYPE_S16:
      ss << "HB_DNN_TENSOR_TYPE_S16";
      break;
    case HB_DNN_TENSOR_TYPE_U16:
      ss << "HB_DNN_TENSOR_TYPE_U16";
      break;
    case HB_DNN_TENSOR_TYPE_F32:
      ss << "HB_DNN_TENSOR_TYPE_F32";
      break;
    case HB_DNN_TENSOR_TYPE_S32:
      ss << "HB_DNN_TENSOR_TYPE_S32";
      break;
    case HB_DNN_TENSOR_TYPE_U32:
      ss << "HB_DNN_TENSOR_TYPE_U32";
      break;
    case HB_DNN_TENSOR_TYPE_F64:
      ss<< "HB_DNN_TENSOR_TYPE_F64";
      break;
    case HB_DNN_TENSOR_TYPE_S64:
      ss << "HB_DNN_TENSOR_TYPE_S64";
      break;
    case HB_DNN_TENSOR_TYPE_U64:
      ss << "HB_DNN_TENSOR_TYPE_U64";
      break;
    case HB_DNN_TENSOR_TYPE_MAX:
      ss << "HB_DNN_TENSOR_TYPE_MAX";
      break;
    default:
      CV_Error(cv::Error::StsBadArg, "Wrong hbDNNTensorLayout value:" + std::to_string(c1));
      break;
  }
  LOG(INFO) << ss.str();
}

void showhbDNNTensorProperties(const hbDNNTensorProperties &c1)
{
  LOG(INFO) << "[hbDNNTensorProperties] List Properties: ";
  
  LOG(INFO) << "  validShape:";
  showhbDNNTensorShape(c1.validShape);
  
  LOG(INFO) << "  alignedShape:";
  showhbDNNTensorShape(c1.alignedShape);

  LOG(INFO) << "  tensorLayout:";
  showhbDNNTensorLayout(hbDNNTensorLayout(c1.tensorLayout));

  LOG(INFO) << "  tensorType:";
  showhbDNNDataType(hbDNNDataType(c1.tensorType));

  LOG(INFO) << "  shift:";
  showhbDNNQuantiShift(c1.shift);

  LOG(INFO) << "  scale:";
  showhbDNNQuantiScale(c1.scale);

  LOG(INFO) << "  quantiType:";
  showhbDNNQuantiType(c1.quantiType);

  LOG(INFO) << "  alignedByteSize: " << c1.alignedByteSize;
}





}

}