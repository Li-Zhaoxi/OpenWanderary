#ifndef PRE_PROCESS_H_
#define PRE_PROCESS_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>



enum VLOG_LEVEL {
  EXAMPLE_SYSTEM = 0,
  EXAMPLE_REPORT = 1,
  EXAMPLE_DETAIL = 2,
  EXAMPLE_DEBUG = 3
};


/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t image2nv12(uint8_t *_bgr_mat, int irows, int icols, void * _data, int input_h, int input_w);
int32_t read_image_2_tensor_as_nv12(cv::Mat &imgC,
                                    hbDNNTensor *input_tensor);

#endif