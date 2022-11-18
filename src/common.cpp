#include "common.h"

std::ostream &operator<< (std::ostream &out, hbDNNQuantiShift &c1)
{
    out << "[hbDNNQuantiShift] shiftLen: " << c1.shiftLen << ", shiftData: " << cv::Mat(1, c1.shiftLen, CV_8UC1, c1.shiftData);
    return out;
}

std::ostream &operator<< (std::ostream &out, hbDNNTensorShape &c1)
{
    out << "[hbDNNTensorShape] dim: " << c1.numDimensions << " [";
    for(int k = 0; k < c1.numDimensions; k++)
    {
        out << c1.dimensionSize[k];
        if (k < c1.numDimensions - 1)
            out << "x";
    }
    out << "]";
    return out;
}

std::ostream &operator<< (std::ostream &out, hbDNNQuantiScale &c1)
{
    out << "[hbDNNQuantiScale] ";
    out << "scaleLen: " << c1.scaleLen << ", scaleData: " << c1.scaleData << " ";
    if (c1.scaleLen != 0 && c1.scaleData)
        out << cv::Mat(1, c1.scaleLen, CV_32FC1, c1.scaleData);
    else 
        out << "[]";
    
    out << ", zeroPointLen: " << c1.zeroPointLen << ", zeroPointData: " << " ";
    if (c1.zeroPointLen != 0 && c1.zeroPointData)
        out << cv::Mat(1, c1.zeroPointLen, CV_8SC1, c1.zeroPointData);
    else 
        out << "[]";

    return out;
}


std::ostream &operator<< (std::ostream &out, hbDNNQuantiType &c1)
{
    out << "[hbDNNQuantiType]: ";
    if (c1 == NONE)
        out << "NONE";
    else if (c1 == SHIFT)
        out << "SHIFT";
    else if (c1 == SCALE)
        out << "SCALE";
    else 
        out << c1;
    return out;
}

hbDNNTensorLayout get_hbDNNTensorLayout(int val)
{
    if (val == HB_DNN_LAYOUT_NHWC)
        return HB_DNN_LAYOUT_NHWC;
    else if (val == HB_DNN_LAYOUT_NCHW)
        return HB_DNN_LAYOUT_NCHW;
    else if (val == HB_DNN_LAYOUT_NONE)
        return HB_DNN_LAYOUT_NONE;
    else
        assert(0);
    return HB_DNN_LAYOUT_NONE;
}

std::ostream &operator<< (std::ostream &out, hbDNNTensorLayout &c1)
{
    out << "[hbDNNTensorLayout]: ";
    if (c1 == HB_DNN_LAYOUT_NHWC )
        out << "HB_DNN_LAYOUT_NHWC ";
    else if (c1 == HB_DNN_LAYOUT_NCHW )
        out << "HB_DNN_LAYOUT_NCHW ";
    else if (c1 == HB_DNN_LAYOUT_NONE )
        out << "HB_DNN_LAYOUT_NONE ";
    else 
        out << c1;
    return out;
}

std::ostream &operator<< (std::ostream &out, hbDNNDataType &c1)
{
    out << "[hbDNNDataType]: ";
    if (c1 == HB_DNN_IMG_TYPE_Y )
        out << "HB_DNN_IMG_TYPE_Y ";
    else if (c1 == HB_DNN_IMG_TYPE_NV12 )
        out << "HB_DNN_IMG_TYPE_NV12 ";
    else if (c1 == HB_DNN_IMG_TYPE_NV12_SEPARATE )
        out << "HB_DNN_IMG_TYPE_NV12_SEPARATE ";
    else if (c1 == HB_DNN_IMG_TYPE_YUV444 )
        out << "HB_DNN_IMG_TYPE_YUV444 ";
    else if (c1 == HB_DNN_IMG_TYPE_RGB )
        out << "HB_DNN_IMG_TYPE_RGB ";
    else if (c1 == HB_DNN_IMG_TYPE_BGR )
        out << "HB_DNN_IMG_TYPE_BGR ";
    else if (c1 == HB_DNN_TENSOR_TYPE_S4 )
        out << "HB_DNN_TENSOR_TYPE_S4 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_U4 )
        out << "HB_DNN_TENSOR_TYPE_U4 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_S8 )
        out << "HB_DNN_TENSOR_TYPE_S8 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_U8 )
        out << "HB_DNN_TENSOR_TYPE_U8 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_F16 )
        out << "HB_DNN_TENSOR_TYPE_F16 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_S16 )
        out << "HB_DNN_TENSOR_TYPE_S16 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_U16 )
        out << "HB_DNN_TENSOR_TYPE_U16 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_F32 )
        out << "HB_DNN_TENSOR_TYPE_F32 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_S32 )
        out << "HB_DNN_TENSOR_TYPE_S32 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_U32 )
        out << "HB_DNN_TENSOR_TYPE_U32 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_F64 )
        out << "HB_DNN_TENSOR_TYPE_F64 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_S64 )
        out << "HB_DNN_TENSOR_TYPE_S64 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_U64 )
        out << "HB_DNN_TENSOR_TYPE_U64 ";
    else if (c1 == HB_DNN_TENSOR_TYPE_MAX )
        out << "HB_DNN_TENSOR_TYPE_MAX ";
    else 
        out << c1;
    return out;
}

std::ostream &operator<< (std::ostream &out, hbDNNTensorProperties &c1)
{
    out << "[hbDNNTensorProperties] List Properties: " << std::endl
        << "  validShape: " << c1.validShape << std::endl
        << "  alignedShape: " << c1.alignedShape << std::endl
        << "  tensorLayout: " << get_hbDNNTensorLayout(c1.tensorLayout) << std::endl
        << "  tensorType: " << c1.tensorType << std::endl
        << "  shift: " << c1.shift << std::endl
        << "  scale: " << c1.scale << std::endl
        << "  quantiType: " << c1.quantiType << std::endl
        << "  alignedByteSize: " << c1.alignedByteSize << std::endl;

    return out;
}

