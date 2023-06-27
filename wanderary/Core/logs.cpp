#include <Core/core.h>

namespace wdr
{
  std::string dtype(cv::InputArray src)
  {
    std::string res = "";
    switch (src.depth())
    {
    case 0:
      res = "CV_8U";
      break;
    case 1:
      res = "CV_8S";
      break;
    case 2:
      res = "CV_16U";
      break;
    case 3:
      res = "CV_16S";
      break;
    case 4:
      res = "CV_32S";
      break;
    case 5:
      res = "CV_32F";
      break;
    case 6:
      res = "CV_64F";
      break;
    case 7:
      res = "CV_16F";
      break;
    default:
      break;
    }

    if (res.length() == 0)
      CV_Error(cv::Error::StsAssert, "Invalid type val: " + std::to_string(src.depth()));

    return res;
  }

  std::string layout(CV_MAT_LAYOUT lay)
  {
    std::string res = "";
    switch (lay)
    {
    case LAYOUT_NONE:
      res = "LAYOUT_NONE";
      break;
    case LAYOUT_A:
      res = "LAYOUT_A";
      break;
    case LAYOUT_AB:
      res = "LAYOUT_AB";
      break;
    case LAYOUT_A1:
      res = "LAYOUT_A1";
      break;
    case LAYOUT_1B:
      res = "LAYOUT_1B";
      break;
    case LAYOUT_ABC:
      res = "LAYOUT_ABC";
      break;
    case LAYOUT_AB1:
      res = "LAYOUT_AB1";
      break;
    case LAYOUT_A1C:
      res = "LAYOUT_A1C";
      break;
    case LAYOUT_1BC:
      res = "LAYOUT_1BC";
      break;
    case LAYOUT_A11:
      res = "LAYOUT_A11";
      break;
    case LAYOUT_1B1:
      res = "LAYOUT_1B1";
      break;
    case LAYOUT_11C:
      res = "LAYOUT_11C";
      break;
    case LAYOUT_ABCD:
      res = "LAYOUT_ABCD";
      break;
    case LAYOUT_ABC1:
      res = "LAYOUT_ABC1";
      break;
    case LAYOUT_AB1D:
      res = "LAYOUT_AB1D";
      break;
    case LAYOUT_A1CD:
      res = "LAYOUT_A1CD";
      break;
    case LAYOUT_1BCD:
      res = "LAYOUT_1BCD";
      break;
    case LAYOUT_AB11:
      res = "LAYOUT_AB11";
      break;
    case LAYOUT_A11D:
      res = "LAYOUT_A11D";
      break;
    case LAYOUT_11CD:
      res = "LAYOUT_11CD";
      break;
    case LAYOUT_A1C1:
      res = "LAYOUT_A1C1";
      break;
    case LAYOUT_1B1D:
      res = "LAYOUT_1B1D";
      break;
    case LAYOUT_1BC1:
      res = "LAYOUT_1BC1";
      break;
    case LAYOUT_A111:
      res = "LAYOUT_A111";
      break;
    case LAYOUT_1B11:
      res = "LAYOUT_1B11";
      break;
    case LAYOUT_11C1:
      res = "LAYOUT_11C1";
      break;
    case LAYOUT_111D:
      res = "LAYOUT_111D";
      break;
    default:
      break;
    }

    if (res.length() == 0)
      CV_Error(cv::Error::StsAssert, "Invalid layout val: " + std::to_string(lay));

    return res;
  }

  bool MatCheck(const cv::Mat &mat, std::string &errmsg, const std::vector<CV_MAT_LAYOUT> &layouts, int channels, const std::vector<int> &dtypes, bool continuous)
  {
    errmsg = "";

    // 检查数据排布
    if (layouts.size())
    {
      std::vector<int> dims = shape(mat);
      const int dimnum = dims.size();
      int range_st = dimnum << 5, range_ed = (dimnum + 1) << 6;
      bool hasfind = false;
      for (auto l : layouts)
      {
        if (l < range_st || l >= range_ed)
          continue;

        switch (l)
        {
        // 2 dims
        case LAYOUT_A1:
          hasfind = dims[1] == 1;
          break;
        case LAYOUT_1B:
          hasfind = dims[0] == 1;
          break;
        // 3dims
        case LAYOUT_AB1:
          hasfind = dims[2] == 1;
          break;
        case LAYOUT_A1C:
          hasfind = dims[1] == 1;
          break;
        case LAYOUT_1BC:
          hasfind = dims[0] == 1;
          break;
        case LAYOUT_A11:
          hasfind = dims[1] == 1 && dims[2] == 1;
          break;
        case LAYOUT_1B1:
          hasfind = dims[0] == 1 && dims[2] == 1;
          break;
        case LAYOUT_11C:
          hasfind = dims[0] == 1 && dims[1] == 1;
          break;
        // 4dims
        case LAYOUT_ABC1:
          hasfind = dims[3] == 1;
          break;
        case LAYOUT_AB1D:
          hasfind = dims[2] == 1;
          break;
        case LAYOUT_A1CD:
          hasfind = dims[1] == 1;
          break;
        case LAYOUT_1BCD:
          hasfind = dims[0] == 1;
          break;
        case LAYOUT_AB11:
          hasfind = dims[2] == 1 && dims[3] == 1;
          break;
        case LAYOUT_A11D:
          hasfind = dims[1] == 1 && dims[2] == 1;
          break;
        case LAYOUT_11CD:
          hasfind = dims[0] == 1 && dims[1] == 1;
          break;
        case LAYOUT_A1C1:
          hasfind = dims[1] == 1 && dims[3] == 1;
          break;
        case LAYOUT_1B1D:
          hasfind = dims[0] == 1 && dims[2] == 1;
          break;
        case LAYOUT_1BC1:
          hasfind = dims[0] == 1 && dims[3] == 1;
          break;
        case LAYOUT_A111:
          hasfind = dims[1] == 1 && dims[2] == 1 && dims[3] == 1;
          break;
        case LAYOUT_1B11:
          hasfind = dims[0] == 1 && dims[2] == 1 && dims[3] == 1;
          break;
        case LAYOUT_11C1:
          hasfind = dims[0] == 1 && dims[1] == 1 && dims[3] == 1;
          break;
        case LAYOUT_111D:
          hasfind = dims[0] == 1 && dims[1] == 1 && dims[2] == 1;
          break;
        }
        if (hasfind)
          break;
      }

      if (!hasfind)
      {
        std::stringstream ss;
        ss << "Shapes [";
        for (auto s : dims)
          ss << s << ",";
        ss << "] not in the format set [";
        for (auto lay : layouts)
          ss << layout(lay) << ",";
        ss << "]; ";

        errmsg += ss.str();
      }
    }

    // 检查通道数
    if (channels > 0)
    {
      if (mat.channels() != channels)
      {
        std::stringstream ss;
        ss << "Channel [" << mat.channels() << "] is not equal to " << channels << "; ";
        errmsg += ss.str();
      }
    }

    // 检查数据类型
    if (dtypes.size() > 0)
    {
      if (!in<int, std::vector<int>>(mat.depth(), dtypes))
      {
        std::stringstream ss;
        ss << "dtype is " << dtype(mat.depth()) << ", not in [";
        for (auto t : dtypes)
          ss << dtype(t) << ", ";
        ss << "]; ";
        errmsg += ss.str();
      }
    }

    // 检查内存连续
    if (continuous)
    {
      if (!mat.isContinuous())
        errmsg += "Memory is not contiguous; ";
    }

    return true;
  }

}