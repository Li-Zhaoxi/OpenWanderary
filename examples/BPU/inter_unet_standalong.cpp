#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <unordered_map>

#include <cnpy/cnpy.h>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/opencv.hpp>


#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      LOG(ERROR) << errmsg << ", error code:" << ret_code; \
      abort();                                               \
    }                                                                \
  } while (0);
void readNets(const std::string &modelpaths,
              hbPackedDNNHandle_t &pPackedNets,
              std::unordered_map<std::string, hbDNNHandle_t> &netsMap);

void readNets(const std::vector<std::string> &modelpaths, 
              hbPackedDNNHandle_t &pPackedNets,
              std::unordered_map<std::string, hbDNNHandle_t> &netsMap);
void readNetProperties(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensorProperties> &properties, bool input);
void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors, bool autopadding = true);
void forward(const hbDNNHandle_t dnn_handle, const std::vector<hbDNNTensor> &inTensors, std::vector<hbDNNTensor> &outTensors, int waiting_time = 0);
void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds);
void releaseTensors(std::vector<hbDNNTensor> &tensors);

// input[0]:
// valid shape: (1,256,256,3,)
// aligned shape: (1,256,256,4,)
// tensor type: HB_DNN_IMG_TYPE_BGR
// tensor layout: HB_DNN_LAYOUT_NHWC
// quanti type: SHIFT
// shift data: 0,0,0,
// output[0]:
// valid shape: (1,2,256,256,)
// aligned shape: (1,2,256,256,)
// tensor type: HB_DNN_TENSOR_TYPE_F32
// tensor layout: HB_DNN_LAYOUT_NCHW
// quanti type: NONE

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // basic infos
  std::string binpath = "examples/modules/unet.bin";
  std::string unetfield = "unet";
  std::string imgpath = "examples/modules/CONIC_161.png";

  // 1. 加载模型
  hbPackedDNNHandle_t pPackedNets = nullptr;
  std::unordered_map<std::string, hbDNNHandle_t> netsMap;
  // readNets({binpath}, pPackedNets, netsMap);
  readNets(binpath, pPackedNets, netsMap);

  // 2. 获取BPU属性，并分配对应BPU内存
  hbDNNHandle_t unet = netsMap[unetfield];
  std::vector<hbDNNTensorProperties> inProperties, outProperties;
  std::vector<hbDNNTensor> inTensors, outTensors;
  readNetProperties(unet, inProperties, true);
  readNetProperties(unet, outProperties, false);
  createTensors(inProperties, inTensors, true);
  createTensors(outProperties, outTensors, false);

  // 3. 将图像数据传入BPU中
  cv::Mat img = cv::imread(imgpath), imgmod, output;
  cv::resize(img, imgmod, cv::Size(256, 256));
  auto data = inTensors[0].sysMem[0].virAddr;
  memcpy(reinterpret_cast<uint8_t *>(data), imgmod.data, 256 * 256 * 3);
  cnpy::npy_save("bpumatin.npy",reinterpret_cast<uint8_t *>(data),{256, 256, 3}, "w");

  HB_CHECK_SUCCESS(hbSysFlushMem(&inTensors[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem cpu->tensor failed");

  // 4. 执行推理
  forward(unet, inTensors, outTensors);

  // 5. 将结果拷拷贝到CPU
  HB_CHECK_SUCCESS(hbSysFlushMem(&outTensors[0].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
        "hbSysFlushMem tensor->cpu failed");
  
  // valid shape: (1,2,256,256,) 这里我就不动态识别大小了
  std::vector<int> dims = {1, 2, 256, 256};
  cv::Mat pred(4, &dims[0], CV_32FC1);
  int dstmemsize = pred.total() * pred.elemSize();
  std::cout << "size: " << dstmemsize << ", " << 1 * 2 * 256 * 256 * 4 << std::endl;
  memcpy(reinterpret_cast<uint8_t *>(pred.data),
         reinterpret_cast<uint8_t *>(outTensors[0].sysMem[0].virAddr),
         dstmemsize);
  cnpy::npy_save("bpumatout.npy",reinterpret_cast<float *>(outTensors[0].sysMem[0].virAddr),{1, 2, 256, 256}, "w");

  std::vector<cv::Mat> batchpreds;
  parseBinarySegmentResult(pred, batchpreds);
  cv::imwrite("outimg.png", batchpreds[0]);
  releaseTensors(inTensors), releaseTensors(outTensors);

  HB_CHECK_SUCCESS(hbDNNRelease(pPackedNets), "hbDNNRelease tensor->cpu failed");
  google::ShutdownGoogleLogging();
  return 0;
}


void readNets(const std::string &modelpaths,
              hbPackedDNNHandle_t &pPackedNets,
              std::unordered_map<std::string, hbDNNHandle_t> &netsMap)
{
  // Init Bin Models from files
  CV_Assert(pPackedNets == nullptr);
  {
    const char* model_name = modelpaths.c_str();
    HB_CHECK_SUCCESS(
      hbDNNInitializeFromFiles(&pPackedNets, &model_name, 1),
      "hbDNNInitializeFromFiles failed");
  }
  // Get All model handles
  netsMap.clear();
  {
    const char **model_name_list;
    int model_count = 0;
    HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, pPackedNets),
      "hbDNNGetModelNameList failed");
    LOG(INFO) << "Input model num: " << model_count << ", Parse model num: " << model_count;

    // Fetch all model handles
    for(int i = 0; i < model_count;i++)
    {
      hbDNNHandle_t tmp;
      const std::string modelname(model_name_list[i]);
      LOG(INFO) << "Fetching the handle of \"" << modelname << "\"";
      HB_CHECK_SUCCESS(hbDNNGetModelHandle(&tmp, pPackedNets, modelname.c_str()),
        "hbDNNGetModelHandle failed");
      netsMap.insert(std::make_pair(modelname, tmp));
    }
  }
}


void readNets(const std::vector<std::string> &modelpaths, 
              hbPackedDNNHandle_t &pPackedNets,
              std::unordered_map<std::string, hbDNNHandle_t> &netsMap)
{
  // Init Bin Models from files
  CV_Assert(pPackedNets == nullptr);
  {
    const int pathnum = modelpaths.size();
    const char **cpaths = new const char*[pathnum];
    for(int k = 0; k < pathnum; k++)
      cpaths[k] = modelpaths[k].c_str();
    HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&pPackedNets, cpaths, pathnum),
      "hbDNNInitializeFromFiles failed");
    delete[] cpaths;
  }

  // Get All model handles
  netsMap.clear();
  {
    const char **model_name_list;
    int model_count = 0;
    HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, pPackedNets),
      "hbDNNGetModelNameList failed");
    LOG(INFO) << "Input model num: " << model_count << ", Parse model num: " << model_count;

    // Fetch all model handles
    for(int i = 0; i < model_count;i++)
    {
      hbDNNHandle_t tmp;
      const std::string modelname(model_name_list[i]);
      LOG(INFO) << "Fetching the handle of \"" << modelname << "\"";
      HB_CHECK_SUCCESS(hbDNNGetModelHandle(&tmp, pPackedNets, modelname.c_str()),
        "hbDNNGetModelHandle failed");
      netsMap.insert(std::make_pair(modelname, tmp));
    }
  }
}

void readNetProperties(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensorProperties> &properties, bool input)
{
  int tensornum = 0;
  if (input)
  {
    HB_CHECK_SUCCESS(hbDNNGetInputCount(&tensornum, dnn_handle), "hbDNNGetInputCount failed");
    LOG(INFO) << "input tensortnum: " << tensornum;
  }
  else
  {
    HB_CHECK_SUCCESS(hbDNNGetOutputCount(&tensornum, dnn_handle), "hbDNNGetOutputCount failed");
    LOG(INFO) << "output tensortnum: " << tensornum;
  }

  properties.resize(tensornum);

  for(int i = 0; i < tensornum; i++)
  {
    auto &usage_property = properties[i];
    if (input)
    {
      HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&usage_property, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");
    }
    else
    {
      HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&usage_property, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    }
  }
}

void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors, bool autopadding)
{
  const int tensornum = properties.size();
  tensors.resize(tensornum);
  for(int i = 0; i < tensornum; i++)
  {
    auto &usage_tensor = tensors[i];
    usage_tensor.properties = properties[i];

    int memSize = usage_tensor.properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");
    
    if (autopadding)
      usage_tensor.properties.alignedShape = usage_tensor.properties.validShape;
  }
}

void forward(const hbDNNHandle_t dnn_handle, const std::vector<hbDNNTensor> &inTensors, std::vector<hbDNNTensor> &outTensors, int waiting_time)
{
  hbDNNInferCtrlParam infer_ctrl_param;
	hbDNNTaskHandle_t task_handle = nullptr;
	HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
	
	auto _poutput_tensors = outTensors.data();
	HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &_poutput_tensors, inTensors.data(),
                              dnn_handle, &infer_ctrl_param), "hbDNNInfer failed");

	// wait task done
	HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, waiting_time), "hbDNNWaitTaskDone failed");
	
	// release task handle
	HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
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

void releaseTensors(std::vector<hbDNNTensor> &tensors)
{
  for(int i = 0; i < tensors.size(); i++)
  {
    HB_CHECK_SUCCESS(hbSysFreeMem(&(tensors[i].sysMem[0])), 
          "hbDNNRelease tensor->cpu failed");
  }
}