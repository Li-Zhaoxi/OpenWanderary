#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <boost/filesystem.hpp>

#include <BPU/bpu.h>
#include <DNN/postproc.h>
#include <gflags/gflags.h>

const std::string binpath = "examples/modules/unet.bin";
const std::string imgpath = "examples/modules/CONIC_161.png";

DEFINE_string(mode, "CAPI", ""); // CAPI/CPPAPI/WDR

int test_c_api(); // 调用BPU的C语言接口开发

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (getuid())
    CV_Error(cv::Error::StsError, "You must use ROOT or SUDO to use these BPU functions.");

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(binpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the model file:  " + binpath);

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(imgpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the image file:  " + imgpath);

  std::string mode = FLAGS_mode;
  LOG(INFO) << "mode: " << mode;

  if (mode == "CAPI")
    test_c_api();
  else
    CV_Error(cv::Error::StsAssert, "Unknown mode: " + mode);

  // // 1. load model and its handle
  // hbPackedDNNHandle_t pPackedNets = nullptr;
  // std::unordered_map<std::string, hbDNNHandle_t> netsMap;
  // wdr::BPU::readNets({binpath}, pPackedNets, netsMap);

  // // 2. Get Tensor Properties and Allocate BPU Memory
  // hbDNNHandle_t hunet = netsMap[unetfield];
  // std::vector<hbDNNTensor> inTensors, outTensors;

  // wdr::BPU::createTensors(hunet, inTensors, true, true);
  // wdr::BPU::createTensors(hunet, outTensors, false, false);

  // // wdr::BPU::showhbDNNTensorProperties(inTensors[0]);
  // // wdr::BPU::showhbDNNTensorProperties(inTensors[0]);

  // // 3. Start Interface
  // //  Read Image->Pre-processing->doInfer->Post-processing->Save Result
  // cv::Mat img = cv::imread(imgpath), imgmod, output;
  // cv::resize(img, imgmod, cv::Size(256, 256));
  // wdr::BPU::bpuMemcpy(imgmod, inTensors[0]); // only one input
  // wdr::BPU::forward(hunet, inTensors, outTensors);
  // wdr::BPU::bpuMemcpy(outTensors[0], output);

  // std::vector<cv::Mat> preds;
  // wdr::parseBinarySegmentResult(output, preds);
  // std::cout << output.size << std::endl;

  // // 4. Save Results;
  // cv::imwrite("pred.png", preds[0]);

  // // 5. Release Tensors and Models
  // wdr::BPU::releaseTensors(inTensors), wdr::BPU::releaseTensors(outTensors);
  // wdr::BPU::releaseNets(pPackedNets);

  google::ShutdownGoogleLogging();
  return 0;
}

int test_c_api()
{
  /**
   * 这里应该对每个BPU函数套用一个HB_CHECK_SUCCESS来检查是否成功执行，比如
   * HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&pPackedNets, cpaths, pathnum), "hbDNNInitializeFromFiles failed");
   * 为了方便理解流程，下面的代码省去了HB_CHECK_SUCCESS
   */

  //////////// 模型加载部分 ////////////
  // 1. 加载BIN模型集
  hbPackedDNNHandle_t packed_dnn_handle; // 模型集合指针
  const char *model_file_name = binpath.c_str();
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);

  // 2. 提取模型集中所有的模型名称
  const char **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
  for (int k = 0; k < model_count; k++) // 输出提取出的所有模型的名称
    LOG(INFO) << "Parsed Model Name: " << std::string(model_name_list[k]);
  // 3. 利用目标模型名提取模型指针
  hbDNNHandle_t dnn_handle; // 模型指针
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  //////////// 输入输出内存分配 ////////////
  // 1. 获取输入/输出Tensor个数
  int input_tensornum = 0, output_tensornum = 0;
  hbDNNGetInputCount(&input_tensornum, dnn_handle);
  hbDNNGetOutputCount(&output_tensornum, dnn_handle);

  // 2. 获取输入/输出Tensor参数
  std::vector<hbDNNTensorProperties> input_properties, output_properties; // 输入/输出Tensor参数
  input_properties.resize(input_tensornum), output_properties.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
    hbDNNGetInputTensorProperties(&input_properties[k], dnn_handle, k);
  for (int k = 0; k < output_tensornum; k++)
    hbDNNGetOutputTensorProperties(&output_properties[k], dnn_handle, k);

  // 3. 利用参数分配Tensor内存
  std::vector<hbDNNTensor> input_tensors, output_tensors; // 输入/输出Tensor
  input_tensors.resize(input_tensornum), output_tensors.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
  {
    const auto &property = input_properties[k];
    input_tensors[k].properties = property;
    hbSysAllocCachedMem(&input_tensors[k].sysMem[0], property.alignedByteSize);
  }
  for (int k = 0; k < output_tensornum; k++)
  {
    const auto &property = output_properties[k];
    output_tensors[k].properties = property;
    hbSysAllocCachedMem(&output_tensors[k].sysMem[0], property.alignedByteSize);
  }

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 加载图像&&图像预处理
  const int imgh = 256, imgw = 256; // 模型的输入Size和分割输出的Size
  cv::Mat img = cv::imread(imgpath);
  // 转换模型时，输入的Tensor排布为HWC，因此不需要进行通道变换
  cv::resize(img, img, cv::Size(imgw, imgh));
  // 由于输入和BPU的Tensor存在不对齐问题，因此需要配置为自动对齐
  auto &tensor = input_tensors[0];
  tensor.properties.alignedShape = tensor.properties.validShape;

  // 2. 预处理数据memcpy至
  memcpy(tensor.sysMem[0].virAddr, img.data, img.total() * img.elemSize());

  // 3. 刷新CPU数据到BPU
  hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

  // 4. 推理模型
  hbDNNTaskHandle_t task_handle = nullptr; // 任务句柄
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  auto ptr_outtensor = output_tensors.data();
  hbDNNInfer(&task_handle, &ptr_outtensor, input_tensors.data(), dnn_handle, &infer_ctrl_param);

  // 5. 等待任务结束
  hbDNNWaitTaskDone(task_handle, 0);

  // 6. 释放任务
  hbDNNReleaseTask(task_handle);

  // 7. 刷新BPU数据到CPU
  hbSysFlushMem(&(output_tensors[0].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  // 8. 从Tensor地址memcpy后处理数据
  std::vector<int> dims = {1, 2, imgh, imgw}; // 模型输出维度
  cv::Mat pred(dims.size(), &dims[0], CV_32FC1);
  memcpy(pred.data, (unsigned char *)output_tensors[0].sysMem[0].virAddr, pred.total() * pred.elemSize());

  // 9. 数据后处理+保存最终分割结果

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 释放内存
  for (auto &input : input_tensors)
    hbSysFreeMem(&(input.sysMem[0]));
  for (auto &output : output_tensors)
    hbSysFreeMem(&(output.sysMem[0]));

  // 2. 释放模型
  hbDNNRelease(packed_dnn_handle);
}
