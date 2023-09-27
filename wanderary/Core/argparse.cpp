#include "core.h"

namespace wdr::argparse
{
  std::vector<std::string> split(const std::string &strlist, const std::string &strsign)
  {
    std::vector<std::string> res;
    if (strlist.length() == 0)
      return res;

    std::string strdata = strlist;
    size_t pos = 0;
    while ((pos = strdata.find(strsign)) != std::string::npos)
    {
      std::string token = strdata.substr(0, pos);
      std::cout << "token: " << token << std::endl;
      if (token.length() == 0)
        CV_Error(cv::Error::StsAssert, "The length of substr is 0. Please check: " + strlist);

      res.push_back(token);
      strdata.erase(0, pos + strsign.length());
    }

    if (strdata.length() > 0)
      res.push_back(strdata);

    return res;
  }
}