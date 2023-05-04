#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>

float quanti_shift(int32_t data, uint32_t shift) {
  return static_cast<float>(data) / static_cast<float>(1 << shift);
}

float quanti_scale(int32_t data, float scale) { return data * scale; }

void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds);


int main(int argc, char **argv) {
  // 第一步加载模型
  hbPackedDNNHandle_t packed_dnn_handle;
  const char* model_file_name= "examples/modules/unet.bin";
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);

  // 第二步获取模型名称
  const char **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

  // 第三步获取dnn_handle
  hbDNNHandle_t dnn_handle;
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // 第四步准备输入数据
  std::string imgname = "examples/modules/CONIC_161.png";
  cv::Mat img = cv::imread(imgname), imgscale;
  cv::resize(img, imgscale, cv::Size(256, 256));
  
  hbDNNTensor input;
  hbDNNTensorProperties input_properties;
  hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
  input.properties = input_properties;
  auto &mem = input.sysMem[0];

  int yuv_length = 256 * 256 * 4;
  hbSysAllocCachedMem(&mem, yuv_length);
  memcpy(mem.virAddr, imgscale.data, 256 * 256 * 3);
  hbSysFlushMem(&mem, HB_SYS_MEM_CACHE_CLEAN);

  // 第五步准备模型输出数据的空间
  int output_count;
  hbDNNGetOutputCount(&output_count, dnn_handle);
  hbDNNTensor *output = new hbDNNTensor[output_count];
  for (int i = 0; i < output_count; i++) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
    int out_aligned_size = output_properties.alignedByteSize;
    hbSysMem &mem = output[i].sysMem[0];
    hbSysAllocCachedMem(&mem, out_aligned_size);
  }

  // 第六步推理模型
  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  hbDNNInfer(&task_handle,
              &output,
              &input,
              dnn_handle,
              &infer_ctrl_param);

  // 第七步等待任务结束
  hbDNNWaitTaskDone(task_handle, 0);
  hbDNNReleaseTask(task_handle);
  //第八步解析模型输出，例子就获取mobilenetv1的top1分类
  hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  float *data = reinterpret_cast< float *>(output->sysMem[0].virAddr);

  cv::Mat label = cv::Mat::zeros(256, 256, CV_8UC1);

  for(int i = 0; i < 256; i++)
  {
    for(int j = 0; j < 256; j++)
    {
      int k = i * 256 + j;
      if (data[k + 256 * 256] > data[k])
        label.at<uchar>(i, j ) = 255;
    }
  }
  cv::imwrite("outimg_demo_label.png", label);
  std::vector<int> dims = {1, 2, 256, 256};
  cv::Mat pred(4, &dims[0], CV_32FC1);
  int dstmemsize = pred.total() * pred.elemSize();
  std::cout << "size: " << dstmemsize << ", " << 1 * 2 * 256 * 256 * 4 << std::endl;
  memcpy(reinterpret_cast<uint8_t *>(pred.data),
         reinterpret_cast<uint8_t *>(data),
         dstmemsize);
  std::vector<cv::Mat> batchpreds;
  parseBinarySegmentResult(pred, batchpreds);
  cv::imwrite("outimg_demo.png", batchpreds[0]);

  
  // 释放内存
  hbSysFreeMem(&(input.sysMem[0]));
  hbSysFreeMem(&(output->sysMem[0]));

  // 释放模型
  hbDNNRelease(packed_dnn_handle);

  return 0;
}


void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds)
{
  // 仅支持CHW，并约束C=2
  CV_Assert(src.rows == -1 && src.cols == -1 && src.channels() == 1 && src.type() == CV_32F);
  CV_Assert(src.size.dims() == 3 || src.size.dims() == 4);
  int b, c, h, w;
  if (src.size.dims() == 3)
    b = 1, c = src.size[0], h = src.size[1], w = src.size[2];
  else
    b = src.size[0], c = src.size[1], h = src.size[2], w = src.size[3];  
  std::cout << "size: " << src.size << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c: " << c << std::endl;
  std::cout << "h: " << h << std::endl;
  std::cout << "w: " << w << std::endl; 
  // 只要通道1大于通道0即可
  CV_Assert(c == 2);
  preds.resize(b);
  for(int i = 0; i < b; i++)
  {
    preds[i].create(h, w, CV_8UC1);
    float *_bdata = ((float*)src.data) + i * c * h * w;
    float *_fdata = _bdata + h * w;
    unsigned char *_label = preds[i].data;
    
    int total_hw = h * w;
    for (int k = 0; k < total_hw; k++)
    {
      *_label = *_fdata > *_bdata ? 255 : 0;
      // std::cout << *_fdata << ", " << *_bdata << ", " << int(*_label) << std::endl;
      _label++, _bdata++, _fdata++;
    }

  }
}
