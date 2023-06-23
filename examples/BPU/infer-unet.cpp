#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <boost/filesystem.hpp>

#include <cnpy/cnpy.h> // 3rdparty

#include <BPU/bpu.h>
#include <DNN/preproc.h>
#include <DNN/postproc.h>
#include <gflags/gflags.h>

const std::string saveroot = "projects/torchdnn/data/unet/";
const std::string binpath = saveroot + "model_output/unet.bin";
const std::string imgpath = saveroot + "mra_img_12.jpg";

DEFINE_string(mode, "class", ""); // CAPI/CPPAPI/WDR

int test_class(); // 调用WDR相关Class实现推理
int test_apis();  // 在C语言接口基础上，测试一些API，Debug用

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  wdr::RootRequired();

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(binpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the model file:  " + binpath);

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(imgpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the image file:  " + imgpath);

  std::string mode = FLAGS_mode;
  LOG(INFO) << "mode: " << mode;

  if (mode == "class")
    test_class();
  else if (mode == "test")
    test_apis();
  else
    CV_Error(cv::Error::StsAssert, "Unknown mode: " + mode);

  google::ShutdownGoogleLogging();
  return 0;
}

int test_class()
{
  const std::string modelname = "unet";
  // 1. 加载模型
  wdr::BPU::BpuNets nets;
  nets.readNets({binpath});
  int idxmode = nets.name2index(modelname);
  LOG(INFO) << "model index: " << idxmode;
  CV_Assert(idxmode >= 0);

  // 2. 加载图像
  cv::Mat img;
  wdr::get_bgr_image(imgpath, img);
  LOG(INFO) << "Finish load bgr image";

  // 3. 内存分配
  wdr::BPU::BpuMats input_mats, output_mats;
  nets.init(idxmode, input_mats, output_mats, true);
  LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();

  // 3. 构造预处理输出，模型输入是256，256
  cv::Mat datain;
  cv::Size modsize = input_mats[0].size(false);
  LOG(INFO) << "Input model size: " << modsize;

  wdr::preprocess_onboard_NHWC(img, modsize.height, modsize.width, datain);
  input_mats[0] << datain; // datain数据拷贝到Tensor里
  input_mats.bpu();        // 更新数据到BPU中
  LOG(INFO) << "Finish preprocess";

  // 4. 模型推理
  cv::Mat dataout;
  nets.forward(idxmode, input_mats, output_mats);
  output_mats.cpu();         // 从BPU中下载数据
  output_mats[0] >> dataout; // 从Tensor里拷出数据到dataout
  LOG(INFO) << "Finish infer";

  // 5. 构造后处理数据，并保存最终预测结果
  std::vector<cv::Mat> preds;
  wdr::parseBinarySegmentResult(dataout, preds);
  for (int k = 0; k < preds.size(); k++)
    cv::imwrite(saveroot + "pred_cpp_wdr_" + std::to_string(k) + ".png", preds[k]);

  // 6. 保存校验数据
  std::string savepath = saveroot + "unet_check_wdrresults.npz";
  LOG(INFO) << "Start saving results";
  cnpy::npz_save(savepath, "datain", (unsigned char *)datain.data, wdr::get_shape(datain), "w");
  cnpy::npz_save(savepath, "dataout", (float *)dataout.data, wdr::get_shape(dataout), "a");
  cnpy::npz_save(savepath, "pred", (unsigned char *)preds[0].data, wdr::get_shape(preds[0]), "a");
  LOG(INFO) << "Finish saving results in " << savepath;

  // 内存释放由代码的析构函数自动完成，无需主动调用

  return 0;
}

int test_apis()
{

  // -----------------模型加载部分--------------------
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
  hbDNNHandle_t dnn_handle; // ※模型指针
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // -----------------输入输出内存分配--------------------
  // 1. 获取输入/输出Tensor个数
  int input_tensornum = 0, output_tensornum = 0;
  hbDNNGetInputCount(&input_tensornum, dnn_handle);
  hbDNNGetOutputCount(&output_tensornum, dnn_handle);
  LOG(INFO) << "input tensor num: " << input_tensornum << ", output tensor num: " << output_tensornum;

  // 2. 获取输入/输出Tensor参数
  std::vector<hbDNNTensorProperties> input_properties, output_properties; // 输入/输出Tensor参数
  input_properties.resize(input_tensornum), output_properties.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
    hbDNNGetInputTensorProperties(&input_properties[k], dnn_handle, k);
  for (int k = 0; k < output_tensornum; k++)
    hbDNNGetOutputTensorProperties(&output_properties[k], dnn_handle, k);

  // 3. 利用参数分配Tensor内存
  std::vector<hbDNNTensor> input_tensors, output_tensors; // ※输入/输出Tensor
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
  LOG(INFO) << "Finish initializing input/output tensors";

  //////// Tensor详细属性信息如下：
  // input[0]:
  //   valid shape: (1,256,256,3,)
  //   aligned shape: (1,256,256,4,)
  //   tensor type: HB_DNN_IMG_TYPE_BGR
  //   tensor layout: HB_DNN_LAYOUT_NHWC
  //   quanti type: SHIFT
  //   shift data: 0,0,0,
  // output[0]:
  //   valid shape: (1,2,256,256,)
  //   aligned shape: (1,2,256,256,)
  //   tensor type: HB_DNN_TENSOR_TYPE_F32
  //   tensor layout: HB_DNN_LAYOUT_NCHW
  //   quanti type: NONE

  // -----------------模型推理：预处理→BPU推理→后处理--------------------
  cv::Mat img, datain, dataout, pred;
  // 1. 加载图像&&图像预处理
  wdr::get_bgr_image(imgpath, img);
  wdr::preprocess_onboard_NHWC(img, 256, 256, datain);
  LOG(INFO) << "Finish preprocess_onboard";

  // 测试Input输入数据拷贝
  auto &tensor = input_tensors[0];
  wdr::BPU::bpuMemcpy(datain, tensor, false);

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
  dataout.create(output_tensors[0].properties.alignedShape.numDimensions,
                 output_tensors[0].properties.alignedShape.dimensionSize, CV_32FC1);
  memcpy(dataout.data, (unsigned char *)output_tensors[0].sysMem[0].virAddr, dataout.total() * dataout.elemSize());
  LOG(INFO) << "Finish infer";

  // 9. 数据后处理+保存最终分割结果
  std::vector<cv::Mat> preds;
  wdr::parseBinarySegmentResult(dataout, preds);
  int offset = pred.size[1] * pred.size[2] * pred.elemSize();
  for (int k = 0; k < preds.size(); k++)
    cv::imwrite(saveroot + "test_cpp_wdr_" + std::to_string(k) + ".png", preds[k]);

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 释放内存
  for (auto &input : input_tensors)
    hbSysFreeMem(&(input.sysMem[0]));
  for (auto &output : output_tensors)
    hbSysFreeMem(&(output.sysMem[0]));

  // 2. 释放模型
  hbDNNRelease(packed_dnn_handle);

  return 0;
}