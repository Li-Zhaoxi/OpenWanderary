#include "FCOS/preprocess.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

int32_t image2nv12(uint8_t *_bgr_mat, int irows, int icols, void * _data, int input_h, int input_w)
{
    // resize
    cv::Mat bgr_mat(irows, icols, CV_8UC3, _bgr_mat);
    cv::Mat mat;
    mat.create(input_h, input_w, bgr_mat.type());
    cv::resize(bgr_mat, mat, mat.size(), 0, 0);
    // convert to YUV420
    if (input_h % 2 || input_w % 2) {
        VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
        return -1;
    }

    cv::Mat yuv_mat;
    cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    // copy y data
    auto data = _data;
    int32_t y_size = input_h * input_w;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    // copy uv data
    int32_t uv_height = input_h / 2;
    int32_t uv_width = input_w / 2;
    uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
    uint8_t *u_data = nv12_data + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }

    return 0;
}

int32_t read_image_2_tensor_as_nv12(cv::Mat &imgC,
                                    hbDNNTensor *input_tensor)
{
    if (imgC.empty()) {
        VLOG(EXAMPLE_SYSTEM) << "image is empty!";
        return -1;
    }
    cv::Mat bgr_mat;
    if (imgC.channels() == 3)
        cv::cvtColor(imgC, bgr_mat, cv::COLOR_GRAY2BGR);
    else
        bgr_mat = imgC;
    
    hbDNNTensor *input = input_tensor;
    hbDNNTensorProperties Properties = input->properties;
    int tensor_id = 0;
    int input_h = Properties.validShape.dimensionSize[1];
    int input_w = Properties.validShape.dimensionSize[2];
    if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) 
    {
        input_h = Properties.validShape.dimensionSize[2];
        input_w = Properties.validShape.dimensionSize[3];
    }

    return image2nv12(bgr_mat.data, bgr_mat.rows, bgr_mat.cols,
                    input->sysMem[0].virAddr, input_h, input_w);
    

    
    // // resize
    // cv::Mat mat;
    // mat.create(input_h, input_w, bgr_mat.type());
    // cv::resize(bgr_mat, mat, mat.size(), 0, 0);
    // // convert to YUV420
    // if (input_h % 2 || input_w % 2) {
    //     VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    //     return -1;
    // }

    // cv::Mat yuv_mat;
    // cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    // uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    // // copy y data
    // auto data = input->sysMem[0].virAddr;
    // int32_t y_size = input_h * input_w;
    // memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    // // copy uv data
    // int32_t uv_height = input_h / 2;
    // int32_t uv_width = input_w / 2;
    // uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
    // uint8_t *u_data = nv12_data + y_size;
    // uint8_t *v_data = u_data + uv_height * uv_width;

    // for (int32_t i = 0; i < uv_width * uv_height; i++) {
    //     *nv12++ = *u_data++;
    //     *nv12++ = *v_data++;
    // }
    // return 0;

}