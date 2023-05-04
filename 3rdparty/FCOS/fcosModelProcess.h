#pragma once
#include <iostream>
#include <vector>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/opencv.hpp>

#include "./method/method_data.h"


class fcosModelProcess
{
private:
  hbPackedDNNHandle_t packed_dnn_handle;
  std::string model_file_name= "./fcos_nv12_to_yuv444_preprocess.bin";
  hbDNNHandle_t dnn_handle;
  int32_t img_width;
  int32_t img_height;

  TensorVector output_tensors;
  TensorVector input_tensors;
  TensorVectorPtr output_ptr;
  TensorVectorPtr input_ptr;

  // 模型推理后，将输出组织在一起，格式为[ *output1 *output2 *output3 ... *output15 ]
  std::vector<void *> outputs_data;
  // 每个输出的内存大小，单位为 uint_8
  std::vector<int32_t> output_data_size;
  std::vector<std::vector<int32_t>> output_data_shap;
  

  int32_t prepareTensor();
  void prepareOutputData();
  int32_t read_image_2_tensor_as_nv12(std::string &image_file);
  int32_t read_mat_bgr_2_tensor_as_nv12(cv::Mat src_mat);

  int32_t read_image_2_tensor_as_bgr(std::string &image_file);
  int32_t read_image_2_tensor_as_bgr(cv::Mat src_mat);
  int32_t releaseMem();

    

public:
  fcosModelProcess(std::string &model_path);
  ~fcosModelProcess();
/**
 * @brief 初始化模型
 * 
 * @return int32_t 
 */
  int32_t initModel();
  /**
   * @brief 准备模型的输入图像，图像格式为jpg
   * 
   * @param jpg_image_file 
   * @param img_width 
   * @param img_height 
   * @return int32_t 
   */
  int32_t prepareImage(std::string &jpg_image_file);

  /**
   * @brief 准备模型的输入图像，图像输入为 cv::Mat
   * 
   * @param src_mat 
   * @return int32_t 
   */
  int32_t prepareImage(cv::Mat src_mat);
  int32_t prepareImage(unsigned char *_imgC, int irows, int icols)
  {
    cv::Mat src_mat(irows, icols, CV_8UC3, _imgC);
    return prepareImage(src_mat);
  }

  int32_t prepareImageBGR(cv::Mat src_mat);
  int32_t prepareImageBGR(unsigned char *_imgC, int irows, int icols)
  {
    cv::Mat src_mat(irows, icols, CV_8UC3, _imgC);
    return prepareImageBGR(src_mat);
  }
  /**
   * @brief 模型推理
   *
   * @return int32_t
   */
  int32_t doInfer();

  /**
   * @brief 模型后处理
   * 
   * @return PerceptionPtr 
   */
  PerceptionPtr postProcess();

  
  int postProcess(std::vector<int> &idx, std::vector<cv::Vec4f> &detbbox, std::vector<float> &score);

  int doPostProcess()
  {
    return postProcess(idxs, detbboxs, scores);
  }
  void UpdateResults(signed int *_idxs, float *_detbboxs, float *_scores, int num)
  {
    for(int i = 0; i < num; i++)
    {
      _idxs[i] = idxs[i];

      _detbboxs[i*4 + 0] = detbboxs[i][0];
      _detbboxs[i*4 + 1] = detbboxs[i][1];
      _detbboxs[i*4 + 2] = detbboxs[i][2];
      _detbboxs[i*4 + 3] = detbboxs[i][3];

      _scores[i] = scores[i];
    }

  }
  std::vector<int> idxs;
  std::vector<cv::Vec4f> detbboxs;
  std::vector<float> scores;
  /**
   * @brief 绘制检测框图像
   * 
   * @param src_img_file 
   * @param dst_img_file 
   * @return int32_t 
   */
  int32_t drawPostImage(std::string src_img_file,std::string dst_img_file,PerceptionPtr results);

  int getOutputsNum() 
  {
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    return output_count;
  }
  int getIdxOutputDim(int idx)
  {
    auto &shape =  output_data_shap[idx];
    return shape.size();
  }
  int copyIdxOutputDim(int idx, signed int *_size)
  {
    int total = 1;
    auto &shape = output_data_shap[idx];
    for (int k = 0; k < shape.size(); k++)
    {
      _size[k] = shape[k];
      total *= shape[k];
    }
    return total;
      
  }
  void copyIdxOutputdata(int idx, float *_data)
  {
    std::vector<void *> *outputs = getOutputsData();
    std::vector<int32_t> *output_size = getOutputsDataSize();
    void *_dst = outputs->at(idx);
    int dst_size = output_size->at(idx);
    memcpy(_data, _dst, dst_size);
  }
  std::vector<void *>* getOutputsData() { return &outputs_data; }
  std::vector<int32_t>* getOutputsDataSize() { return &output_data_size; }
  std::vector<std::vector<int32_t>>* getOutputsDataShap() { return &output_data_shap; }

};


void drawPostImage(unsigned char *_imgC, int rows, int cols, int bboxnum, signed int *_idx, float *_detbbox, signed int *_colors, int colornum, int thickness);
void drawPostImage(cv::Mat &imgC, std::vector<int> &idx, std::vector<cv::Vec4f> &detbbox, std::vector<cv::Scalar> &colors, int thickness);