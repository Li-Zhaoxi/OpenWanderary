#include <BPU/bpu.h>


namespace wdr
{

namespace BPU
{
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

void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors)
{
  const int tensornum = properties.size();
  tensors.resize(tensornum);
  for(int i = 0; i < tensornum; i++)
  {
    auto &usage_tensor = tensors[i];
    usage_tensor.properties = properties[i];

    int memSize = usage_tensor.properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");
  }
}

void bpuMemcpy(hbDNNTensor &dst, const uint8_t *src, int memsize)
{
  int memSize = dst.properties.alignedByteSize;
  if (memsize < 0)
    memsize = memSize;
  else
    CV_Assert(memSize >= memsize);
  auto data = dst.sysMem[0].virAddr;

	memcpy(reinterpret_cast<uint8_t *>(data), src, memsize);
	HB_CHECK_SUCCESS(hbSysFlushMem(&dst.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem cpu->tensor failed");
}

void bpuMemcpy(uint8_t *dst, hbDNNTensor &src, int memsize)
{
  int memSize = src.properties.alignedByteSize;
  if (memsize < 0)
    memsize = memSize;
  else
    CV_Assert(memSize >= memsize);

  HB_CHECK_SUCCESS(hbSysFlushMem(&src.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
        "hbSysFlushMem tensor->cpu failed");
  
  auto data = src.sysMem[0].virAddr;
  memcpy(dst, reinterpret_cast<uint8_t *>(data), memsize);
}

void bpuMemcpy(const cv::Mat &src, hbDNNTensor &dst)
{
  // dst tensor infos
  auto &property = dst.properties;
  auto &alignedShape = property.alignedShape;
  int alignedByteSize = property.alignedByteSize;

}

void bpuMemcpy(hbDNNTensor &src, cv::Mat &dst)
{
  HB_CHECK_SUCCESS(hbSysFlushMem(&src.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
        "hbSysFlushMem tensor->cpu failed");
  
  auto &property = src.properties;
  auto &alignedShape = property.alignedShape;
  int alignedByteSize = property.alignedByteSize;

  // allocate data
  std::vector<int> dims(alignedShape.numDimensions);
  for(int k = 0; k < alignedShape.numDimensions; k++)
    dims[k] = alignedShape.dimensionSize[k];
  switch(property.tensorType)
  {
    case HB_DNN_TENSOR_TYPE_S8:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_8S, 1));
      break;
    case HB_DNN_TENSOR_TYPE_U8:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_8U, 1));
      break;
    case HB_DNN_TENSOR_TYPE_S16:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_16S, 1));
      break;
    case HB_DNN_TENSOR_TYPE_U16:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_16U, 1));
      break;
    case HB_DNN_TENSOR_TYPE_F32:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_32F, 1));
      break;
    case HB_DNN_TENSOR_TYPE_S32:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_32S, 1));
      break;
    case HB_DNN_TENSOR_TYPE_F64:
      dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_64F, 1));
      break;
    default:
      LOG(ERROR) << "Unsupport type: ";
      showhbDNNDataType(hbDNNDataType(property.tensorType));
      std::abort();
      break;
  }

  // copy date
  int dstmemsize = dst.total() * dst.elemSize();
  CV_Assert(dstmemsize == alignedByteSize);

  memcpy(reinterpret_cast<uint8_t *>(dst.data),
         reinterpret_cast<uint8_t *>(src.sysMem[0].virAddr),
         dstmemsize);
}

void releaseNets(hbPackedDNNHandle_t &pPackedNets)
{
  HB_CHECK_SUCCESS(hbDNNRelease(pPackedNets), "hbDNNRelease tensor->cpu failed");
  pPackedNets = nullptr;
}

void releaseTensors(std::vector<hbDNNTensor> &tensors)
{
  for(int i = 0; i < tensors.size(); i++)
  {
    HB_CHECK_SUCCESS(hbSysFreeMem(&(tensors[i].sysMem[0])), 
          "hbDNNRelease tensor->cpu failed");
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

} // end BPU


// BPUModels::BPUModels(const std::vector<std::string> &model_paths)
// {
//   // Init Bin Models from files
//   {
//     const int pathnum = model_paths.size();
//     char **cpaths = new char*[pathnum];
//     for(int k = 0; k < pathnum; k++)
//       cpaths[k] = model_paths[k].c_str();
//     HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, cpaths, pathnum),
//       "hbDNNInitializeFromFiles failed");
//     delete[] cpaths;
//   }

  
  

//   const char **model_name_list;
//   int model_count = 0;
//   HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
//     "hbDNNGetModelNameList failed");
//   LOG(INFO) << "Input model num: " << model_num << ", Parse model num: " << model_count;
  
//   // Fetch all model handles
//   for(int i = 0; i < model_count;i++)
//   {
//     hbDNNHandle_t tmp;
//     HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[i]),
//       "hbDNNGetModelHandle failed");
    
//     LOG(INFO) << "Finish fetching the handle of " << std::string(model_name_list[i]);
//     dnn_handles.push_back(tmp);
//   }
  
//   // Prepare Input && Output Tensors
//   all_intensor.resize(model_count), all_outtensor.resize(model_count);


  
//   // BPUModule::prepareInputOutputTensor(dnn_handle, this->input_tensors, true);
//   // BPUModule::prepareInputOutputTensor(dnn_handle, this->output_tensors, false);
// }	



// void BPUModule::prepareInputOutputTensor(const hbDNNHandle_t &dnn_handle, std::vector<hbDNNTensor> &tensors, bool input)
// {
//   int tensornum = 0;
//   if (input)
//   {
//     HB_CHECK_SUCCESS(hbDNNGetInputCount(&tensornum, dnn_handle), "hbDNNGetInputCount failed");
//     LOG(INFO) << "input tensortnum: " << tensornum;
//   }
//   else
//   {
//     HB_CHECK_SUCCESS(hbDNNGetOutputCount(&tensornum, dnn_handle), "hbDNNGetOutputCount failed");
//     LOG(INFO) << "output tensortnum: " << tensornum;
//   }

//   tensors.resize(tensornum);

//   for(int i = 0; i < tensornum; i++)
//   {
//     auto &usage_tensor = tensors[i];
//     if (input)
//     {
//       HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&usage_tensor.properties, dnn_handle, i),
//         "hbDNNGetInputTensorProperties failed");
//       // std::cout << "input_tensors: " << std::endl;
//       // BPUModule::showhbDNNTensorProperties(usage_tensor.properties);
//     }
//     else
//     {
//       HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&usage_tensor.properties, dnn_handle, i),
//         "hbDNNGetOutputTensorProperties failed");
//       // std::cout << "output_tensors: " << std::endl;
//       // BPUModule::showhbDNNTensorProperties(usage_tensor.properties);
//     }

//     auto &usage_properties = usage_tensor.properties;

//     int memSize = usage_properties.alignedByteSize;
//     HB_CHECK_SUCCESS(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");
//   }

// }







} // end wdr
