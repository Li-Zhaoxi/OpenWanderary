#include <iostream>
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      LOG(ERROR) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);


std::ostream &operator<< (std::ostream &out, hbDNNQuantiShift &c1);
std::ostream &operator<< (std::ostream &out, hbDNNTensorShape &c1);
std::ostream &operator<< (std::ostream &out, hbDNNQuantiScale &c1);
std::ostream &operator<< (std::ostream &out, hbDNNQuantiType &c1);
std::ostream &operator<< (std::ostream &out, hbDNNTensorLayout &c1);
std::ostream &operator<< (std::ostream &out, hbDNNDataType &c1);

std::ostream &operator<< (std::ostream &out, hbDNNTensorProperties &c1);