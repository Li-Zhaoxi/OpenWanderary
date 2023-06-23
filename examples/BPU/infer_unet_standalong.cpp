#include <iostream>
#include <vector>
#include <string>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <cnpy/cnpy.h> // 3rdparty
#include <opencv2/opencv.hpp>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>

DEFINE_string(mode, "", "");

#define HB_CHECK_SUCCESS(value, errmsg)                    \
  do                                                       \
  {                                                        \
    /*value can be call of function*/                      \
    auto ret_code = value;                                 \
    if (ret_code != 0)                                     \
    {                                                      \
      LOG(ERROR) << errmsg << ", error code:" << ret_code; \
      abort();                                             \
    }                                                      \
  } while (0);

// 获取Tensor的宽高信息[validShape，非对齐宽高]
cv::Size get_hw(hbDNNTensorProperties &pro)
{
  CV_Assert(pro.validShape.numDimensions == 4);
  if (pro.tensorLayout == HB_DNN_LAYOUT_NCHW)
    return cv::Size(pro.validShape.dimensionSize[3], pro.validShape.dimensionSize[2]);
  else
    return cv::Size(pro.validShape.dimensionSize[2], pro.validShape.dimensionSize[1]);
}

void get_bgr_image(const std::string &imgpath, cv::Mat &img)
{
  // Python: img = cv2.imread(imgpath)
  img = cv::imread(imgpath);

  // Python: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  if (img.channels() == 1)
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
}

void preprocess_onboard(const cv::Mat img, int modelh, int modelw, cv::Mat &datain)
{
  cv::Mat tmp;
  // Python: img = cv2.resize(img, (modelw, modelh))
  cv::resize(img, tmp, cv::Size(modelw, modelh));

  // Python: img = np.expand_dims(img, 0)
  // Python: img = np.ascontiguousarray(img)
  std::vector<int> dims = {1, tmp.rows, tmp.cols, tmp.channels()};
  datain.create(dims.size(), dims.data(), CV_MAKETYPE(img.depth(), 1));

  memcpy(datain.data, tmp.data, tmp.total() * tmp.elemSize());
}

void postprocess(const cv::Mat outputs, cv::Mat &pred)
{
  // 格式检查：保证outputs的维度是[b,2,h,w]，数据类型为float
  CV_Assert(outputs.size.dims() == 4 && outputs.channels() == 1 && outputs.type() == CV_32F);
  int b = outputs.size[0], c = outputs.size[1], h = outputs.size[2], w = outputs.size[3];
  CV_Assert(c == 2);

  // Python: y_list = softmax(outputs, axis = 1)[:, 1, :, :]
  // Python：y_list = (y_list > 0.5).astype(np.uint8) * 255
  // 这里可以简化为，比较[:, 1, :, :]和[:, 0, :, :]的大小，若前景大于背景，Label给255
  std::vector<int> dims = {b, h, w};
  pred.create(dims.size(), dims.data(), CV_8UC1);
  for (int i = 0; i < b; i++)
  {
    float *_bdata = ((float *)outputs.data) + i * c * h * w; // 背景指针
    float *_fdata = _bdata + h * w;                          // 前景指针

    unsigned char *_label = pred.data + i * h * w;

    int total_hw = h * w;
    for (int k = 0; k < total_hw; k++, _label++, _bdata++, _fdata++)
      *_label = *_fdata > *_bdata ? 255 : 0;
  }
}

void infer_unet();
void check_all();

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (getuid())
    CV_Error(cv::Error::StsError, "You must use ROOT or SUDO to use these BPU functions.");

  std::string mode = FLAGS_mode;
  std::cout << "mode: " << mode << std::endl;

  if (mode == "infer")
    infer_unet();
  else if (mode == "check")
    check_all();
  else
    CV_Error(cv::Error::StsAssert, "Unknown mode: " + mode);

  google::ShutdownGoogleLogging();
  return 0;
}

void infer_unet()
{
  /**
   * 这里应该对每个BPU函数套用一个HB_CHECK_SUCCESS来检查是否成功执行，比如
   * HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&pPackedNets, cpaths, pathnum), "hbDNNInitializeFromFiles failed");
   * 为了方便理解流程，下面的代码省去了HB_CHECK_SUCCESS
   */
  std::string dataroot = "projects/torchdnn/data/unet/";
  std::string binpath = dataroot + "model_output/unet.bin";
  std::string imgpath = dataroot + "mra_img_12.jpg";

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
  get_bgr_image(imgpath, img);
  cv::Size tensorhw = get_hw(input_tensors[0].properties);
  LOG(INFO) << "Loaded img size: " << img.size() << ", target size: " << tensorhw;
  preprocess_onboard(img, tensorhw.height, tensorhw.width, datain);
  LOG(INFO) << "Finish preprocess_onboard";

  // 由于输入和BPU的Tensor存在不对齐问题，因此需要配置为自动对齐
  // input_tensors[0]: valid shape: (1,256,256,3,)，aligned shape: (1,256,256,4,)
  auto &tensor = input_tensors[0];
  tensor.properties.alignedShape = tensor.properties.validShape;

  // 2. 预处理数据memcpy至BPU
  memcpy(tensor.sysMem[0].virAddr, datain.data, datain.total() * datain.elemSize());

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
  postprocess(dataout, pred);
  int offset = pred.size[1] * pred.size[2] * pred.elemSize();
  for (int k = 0; k < pred.size[0]; k++)
  {
    cv::Mat batchpred(pred.size[1], pred.size[2], CV_8UC1, pred.data + k * offset);
    cv::imwrite(dataroot + "pred_bin_cpp_b" + std::to_string(k) + ".png", batchpred);
  }

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 释放内存
  for (auto &input : input_tensors)
    hbSysFreeMem(&(input.sysMem[0]));
  for (auto &output : output_tensors)
    hbSysFreeMem(&(output.sysMem[0]));

  // 2. 释放模型
  hbDNNRelease(packed_dnn_handle);
}

std::vector<size_t> get_shape(cv::Mat &mat)
{
  CV_Assert(mat.rows < 0);

  std::vector<size_t> shapes;
  for (int k = 0; k < mat.size.dims(); k++)
    shapes.push_back(mat.size[k]);

  return shapes;
}

std::vector<int> get_dims(const cnpy::NpyArray &arr)
{
  std::vector<int> dims;
  for (auto dim : arr.shape)
    dims.push_back(dim);
  return dims;
}

void check_preprocess(const cnpy::NpyArray &arr_image, cv::Mat &datain)
{
  // Load image
  cv::Mat img;
  CV_Assert(arr_image.shape.size() == 3);
  img.create(arr_image.shape[0], arr_image.shape[1], CV_MAKETYPE(CV_8U, arr_image.shape[2]));
  memcpy(img.data, arr_image.data<unsigned char>(), img.total() * img.elemSize());

  // Get datain
  preprocess_onboard(img, 256, 256, datain);
}

void check_postprocess(const cnpy::NpyArray &arr_dataout, cv::Mat &pred)
{
  // Load dataout
  cv::Mat dataout;
  std::vector<int> dims = get_dims(arr_dataout);
  dataout.create(dims.size(), dims.data(), CV_32FC1);
  memcpy(dataout.data, arr_dataout.data<unsigned char>(), dataout.total() * dataout.elemSize());

  // Get pred
  postprocess(dataout, pred);
}

void check_infer(const std::string &binpath, const cnpy::NpyArray &arr_datain, cv::Mat &dataout)
{
  // Load datain
  cv::Mat datain;
  CV_Assert(arr_datain.shape.size() == 4);
  std::vector<int> dims = get_dims(arr_datain);
  datain.create(dims.size(), dims.data(), CV_8UC1);
  memcpy(datain.data, arr_datain.data<unsigned char>(), datain.total() * datain.elemSize());

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

  // -----------------模型推理：预处理→BPU推理→后处理--------------------
  auto &tensor = input_tensors[0];
  tensor.properties.alignedShape = tensor.properties.validShape;

  // 2. 预处理数据memcpy至BPU
  memcpy(tensor.sysMem[0].virAddr, datain.data, datain.total() * datain.elemSize());

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

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 释放内存
  for (auto &input : input_tensors)
    hbSysFreeMem(&(input.sysMem[0]));
  for (auto &output : output_tensors)
    hbSysFreeMem(&(output.sysMem[0]));

  // 2. 释放模型
  hbDNNRelease(packed_dnn_handle);
}

void check_all()
{
  std::string dataroot = "projects/torchdnn/data/unet/";
  std::string npzpath = dataroot + "unet_checkstage2.npz";
  std::string binpath = dataroot + "model_output/unet.bin";
  std::string savepath = dataroot + "unet_checkcppresults.npz";

  // 加载各阶段理论值npz文件
  cnpy::npz_t datanpz = cnpy::npz_load(npzpath);

  cv::Mat datain, dataout, pred;

  // (1) 预处理校验过程：输入理论图像数据，返回预处理结果
  LOG(INFO) << "Start check_preprocess";
  // 通过字符串可直接访问npz中的数据，返回cnpy::NpyArray
  check_preprocess(datanpz["image"], datain);

  // (2) 后处理校验过程：输入推理理论输出，返回后处理预测结果
  LOG(INFO) << "Start check_postprocess";
  check_postprocess(datanpz["dataout"], pred);

  // (3) 推理校验过程：输入推理所需的理论值，返回推理结果
  LOG(INFO) << "Start check_infer";
  check_infer(binpath, datanpz["datain"], dataout);

  // 保存各个阶段的输出值到npz文件
  LOG(INFO) << "Start saving results";
  cnpy::npz_save(savepath, "datain", (unsigned char *)datain.data, get_shape(datain), "w");
  cnpy::npz_save(savepath, "dataout", (float *)dataout.data, get_shape(dataout), "a");
  cnpy::npz_save(savepath, "pred", (unsigned char *)pred.data, get_shape(pred), "a");
  LOG(INFO) << "Finish saving results in " << savepath;
}