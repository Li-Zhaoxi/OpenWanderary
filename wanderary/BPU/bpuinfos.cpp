#include <BPU/bpu.h>
#include <sstream>
#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>
using nlohmann::json;

namespace wdr
{

  namespace BPU
  {
    std::string formathbDNNQuantiShift(const hbDNNQuantiShift &c1)
    {
      std::stringstream ss;
      ss << "[hbDNNQuantiShift] shiftLen: " << c1.shiftLen << ", shiftData: "
         << cv::Mat(1, c1.shiftLen, CV_8UC1, c1.shiftData);
      return ss.str();
    }

    std::string formathbDNNTensorShape(const hbDNNTensorShape &c1)
    {
      std::stringstream ss;
      ss << "[hbDNNTensorShape] dim: " << c1.numDimensions << " [";
      for (int k = 0; k < c1.numDimensions; k++)
      {
        ss << c1.dimensionSize[k];
        if (k < c1.numDimensions - 1)
          ss << "x";
      }
      ss << "]";
      return ss.str();
    }

    std::string formathbDNNQuantiScale(const hbDNNQuantiScale &c1)
    {
      std::stringstream ss;
      ss << "[hbDNNQuantiScale] ";
      ss << "scaleLen: " << c1.scaleLen << ", scaleData: " << c1.scaleData << " ";
      if (c1.scaleLen != 0 && c1.scaleData)
        ss << cv::Mat(1, c1.scaleLen, CV_32FC1, c1.scaleData);
      else
        ss << "[]";

      ss << ", zeroPointLen: " << c1.zeroPointLen << ", zeroPointData: "
         << " ";
      if (c1.zeroPointLen != 0 && c1.zeroPointData)
        ss << cv::Mat(1, c1.zeroPointLen, CV_8SC1, c1.zeroPointData);
      else
        ss << "[]";

      return ss.str();
    }

    std::string formathbDNNQuantiType(const hbDNNQuantiType &c1)
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
      return ss.str();
    }

    std::string formathbDNNTensorLayout(const hbDNNTensorLayout &c1)
    {
      std::stringstream ss;
      ss << "[hbDNNTensorLayout]: ";
      switch (c1)
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

      return ss.str();
    }

    std::string formathbDNNDataType(const hbDNNDataType &c1)
    {
      std::stringstream ss;
      ss << "[hbDNNDataType]: ";
      switch (c1)
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
        ss << "HB_DNN_TENSOR_TYPE_F64";
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
      return ss.str();
    }

    json packhbDNNTensorProperties(const hbDNNTensorProperties &c1)
    {
      json res;
      res["validShape"] = formathbDNNTensorShape(c1.validShape);
      res["alignedShape"] = formathbDNNTensorShape(c1.alignedShape);

      res["tensorLayout"] = formathbDNNTensorLayout(hbDNNTensorLayout(c1.tensorLayout));
      res["tensorType"] = formathbDNNDataType(hbDNNDataType(c1.tensorType));

      res["shift"] = formathbDNNQuantiShift(c1.shift);
      res["scale"] = formathbDNNQuantiScale(c1.scale);

      res["quantiType"] = formathbDNNQuantiType(c1.quantiType);

      res["quantiType"] = c1.alignedByteSize;

      return res;
    }

    std::string formathbDNNTensorProperties(const hbDNNTensorProperties &c1)
    {
      std::stringstream ss;
      json packedinfos = packhbDNNTensorProperties(c1);

      ss << "[hbDNNTensorProperties] List Properties: " << std::endl;
      ss << packedinfos.dump(2);

      return ss.str();
    }

    std::ostream &operator<<(std::ostream &out, const NetIOInfo &c1)
    {
      json packedtensors;
      for (int k = 0; k < c1.size(); k++)
        packedtensors[k] = packhbDNNTensorProperties(c1.infos[k]);

      out << "[Input/Output] All Tensor Properties: " << std::endl;
      out << packedtensors.dump(2);

      return out;
    }

    std::ostream &operator<<(std::ostream &out, const NetInfos &c1)
    {
      json packedtensors;
      packedtensors["name"] = c1.modelname;
      { // packed input tensors;
        json tmp;
        for (int k = 0; k < c1.input_infos.size(); k++)
          tmp[k] = packhbDNNTensorProperties(c1.input_infos.infos[k]);
        packedtensors["Input"] = tmp;
      }
      { // packed output tensors;
        json tmp;
        for (int k = 0; k < c1.output_infos.size(); k++)
          tmp[k] = packhbDNNTensorProperties(c1.output_infos.infos[k]);
        packedtensors["Output"] = tmp;
      }

      out << "[Net] All Input && Output Tensor Properties: " << std::endl;
      out << packedtensors.dump(2);

      return out;
    }

  }

}

std::ostream &operator<<(std::ostream &out, const hbDNNTensorProperties &c1)
{
  out << wdr::BPU::formathbDNNTensorProperties(c1);
  return out;
}
