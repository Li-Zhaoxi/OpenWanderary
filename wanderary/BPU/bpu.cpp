#include <BPU/bpu.h>
#include <Core/core.h>

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
        const char **cpaths = new const char *[pathnum];
        for (int k = 0; k < pathnum; k++)
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
        // LOG(INFO) << "Input model num: " << model_count << ", Parse model num: " << model_count;

        // Fetch all model handles
        for (int i = 0; i < model_count; i++)
        {
          hbDNNHandle_t tmp;
          const std::string modelname(model_name_list[i]);
          // LOG(INFO) << "Fetching the handle of \"" << modelname << "\"";
          HB_CHECK_SUCCESS(hbDNNGetModelHandle(&tmp, pPackedNets, modelname.c_str()),
                           "hbDNNGetModelHandle failed");
          netsMap.insert(std::make_pair(modelname, tmp));
        }
      }
    }

    void releaseNets(hbPackedDNNHandle_t &pPackedNets)
    {
      HB_CHECK_SUCCESS(hbDNNRelease(pPackedNets), "hbDNNRelease tensor->cpu failed");
      pPackedNets = nullptr;
    }

    void readNetProperties(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensorProperties> &properties, bool input)
    {
      int tensornum = 0;
      if (input)
      {
        HB_CHECK_SUCCESS(hbDNNGetInputCount(&tensornum, dnn_handle), "hbDNNGetInputCount failed");
        // LOG(INFO) << "input tensortnum: " << tensornum;
      }
      else
      {
        HB_CHECK_SUCCESS(hbDNNGetOutputCount(&tensornum, dnn_handle), "hbDNNGetOutputCount failed");
        // LOG(INFO) << "output tensortnum: " << tensornum;
      }

      properties.resize(tensornum);

      for (int i = 0; i < tensornum; i++)
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

    void shape(const hbDNNTensorProperties &property, TensorSize &tensorshape, bool aligned)
    {
      if (aligned)
        tensorshape.create(property.alignedShape.numDimensions, property.alignedShape.dimensionSize);
      else
        tensorshape.create(property.validShape.numDimensions, property.validShape.dimensionSize);
    }

    void shape(cv::InputArray src, TensorSize &cvshape)
    {
      cv::Mat mat = src.getMat();
      if (mat.rows < 0 || mat.cols < 0)
        cvshape.create(mat.size.dims(), mat.size.p);
      else
      {
        cvshape.clear();
        cvshape.push_back(mat.rows);
        cvshape.push_back(mat.cols);
        cvshape.push_back(mat.channels());
      }
    }

    void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors, bool autopadding)
    {
      const int tensornum = properties.size();
      tensors.resize(tensornum);
      for (int i = 0; i < tensornum; i++)
      {
        auto &usage_tensor = tensors[i];
        usage_tensor.properties = properties[i];

        int memSize = usage_tensor.properties.alignedByteSize;
        HB_CHECK_SUCCESS(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");

        if (autopadding)
          usage_tensor.properties.alignedShape = usage_tensor.properties.validShape;
      }
    }

    void createTensors(const hbDNNTensorProperties &property, hbDNNTensor &bputensor)
    {
      bputensor.properties = property;
      int memSize = bputensor.properties.alignedByteSize;
      HB_CHECK_SUCCESS(hbSysAllocCachedMem(&bputensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");
    }

    void createTensors(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensor> &tensors, bool input, bool autopadding)
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

      tensors.resize(tensornum);
      for (int i = 0; i < tensornum; i++)
      {
        auto &usage_tensor = tensors[i];

        if (input)
        {
          HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&usage_tensor.properties, dnn_handle, i),
                           "hbDNNGetInputTensorProperties failed");
        }
        else
        {
          HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&usage_tensor.properties, dnn_handle, i),
                           "hbDNNGetOutputTensorProperties failed");
        }

        int memSize = usage_tensor.properties.alignedByteSize;
        HB_CHECK_SUCCESS(hbSysAllocCachedMem(&usage_tensor.sysMem[0], memSize), "hbSysAllocCachedMem failed");

        if (autopadding)
          usage_tensor.properties.alignedShape = usage_tensor.properties.validShape;
      }
    }

    void flushBPU(hbDNNTensor &dst, bool upload)
    {
      if (upload)
      {
        HB_CHECK_SUCCESS(hbSysFlushMem(&dst.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
                         "hbSysFlushMem cpu->tensor failed");
      }
      else
      {
        HB_CHECK_SUCCESS(hbSysFlushMem(&dst.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
                         "hbSysFlushMem tensor->cpu failed");
      }
    }

    void alignMemory(const unsigned char *src, const TensorSize &srcshape, unsigned char *dst, TensorSize &dstshape, int elementsize)
    {
      CV_Error(cv::Error::StsAssert, "alignMemory exists bugs and is currently not supported for use.");
      if (srcshape <= dstshape)
      {
        // LOG(INFO) << "srcshape <= dstshape";
        int low1 = srcshape[0], low2 = srcshape[1], low3 = srcshape[2], low4 = srcshape[3];
        int high1 = dstshape[0], high2 = dstshape[1], high3 = dstshape[2], high4 = dstshape[3];
        int e1 = high1 - low1, e2 = high2 - low2, e3 = high3 - low3, e4 = high4 - low4;

        LOG(INFO) << "low1: " << low1 << ", low2: " << low2 << ", low3: " << low3 << ", low4: " << low4;
        LOG(INFO) << "e1: " << e1 << ", e2: " << e2 << ", e3: " << e3 << ", e4: " << e4;
        LOG(INFO) << "high1: " << high1 << ", high2: " << high2 << ", high3: " << high3 << ", high4: " << high4;
        // int countsrc = 0, countdst = 0;
        // memset(dst, 0, high1 * high2 * high3 * high4);
        for (int i1 = 0; i1 < low1; i1++)
        {
          for (int i2 = 0; i2 < low2; i2++)
          {
            for (int i3 = 0; i3 < low3; i3++)
            {
              for (int i4 = 0; i4 < low4; i4++)
              {
                *dst = *src;
                dst++, src++;
                // *(dst++) = *(src++);
                // countsrc++, countdst++;
                // LOG(INFO) << "[" << i1 << "," << i2 << "," << i3 << "," << i4 << "]: " << countsrc << ", " << countdst;
              }
              dst += e4;
              // countdst += e4;
            }
            dst += e3 * (high4);
            // countdst += e3 * (high4);
          }
          dst += e2 * (high3 * high4);
          // countdst += e2 * (high3 * high4);
        }
        // LOG(INFO) << "finish";
      }
      else if (srcshape >= dstshape)
      {
        int low1 = dstshape[0], low2 = dstshape[1], low3 = dstshape[2], low4 = dstshape[3];
        int high1 = srcshape[0], high2 = srcshape[1], high3 = srcshape[2], high4 = srcshape[3];
        int e1 = high1 - low1, e2 = high2 - low2, e3 = high3 - low3, e4 = high4 - low4;

        for (int i1 = 0; i1 < low1; i1++)
        {
          for (int i2 = 0; i2 < low2; i2++)
          {
            for (int i3 = 0; i3 < low3; i3++)
            {
              for (int i4 = 0; i4 < low4; i4++)
                *(dst++) = *(src++);
              src += e4;
            }
            src += e3 * (high4);
          }
          src += e2 * (high3 * high4);
        }
      }
      else
      {
        std::stringstream ss;
        ss << "Invalid Shapes. Src shape: " << srcshape << ", dst shape: " << dstshape;
        CV_Error(cv::Error::StsAssert, ss.str());
      }
    }

    void bpuMemcpy(hbDNNTensor &dst, const uint8_t *src, int memsize, bool flush)
    {
      int memSize = dst.properties.alignedByteSize;
      if (memsize < 0)
        memsize = memSize;
      else
        CV_Assert(memSize >= memsize);

      auto data = dst.sysMem[0].virAddr;
      memcpy(reinterpret_cast<uint8_t *>(data), src, memsize);
      if (flush)
        flushBPU(dst, true);
    }

    void bpuMemcpy(uint8_t *dst, hbDNNTensor &src, int memsize, bool flush)
    {
      int memSize = src.properties.alignedByteSize;
      if (memsize < 0)
        memsize = memSize;
      else
        CV_Assert(memSize >= memsize);

      if (flush)
        flushBPU(src, false);

      auto data = src.sysMem[0].virAddr;
      memcpy(dst, reinterpret_cast<uint8_t *>(data), memsize);
    }

    void bpuMemcpy(cv::InputArray src, hbDNNTensor &dst, bool flush)
    {
      // 1 保证使用的数据是连续的
      cv::Mat mat;
      const auto &property = dst.properties;
      {
        cv::Mat tmp;
        if (src.rows() > 0 && src.cols() > 0 && property.tensorLayout != HB_DNN_LAYOUT_NHWC)
          hwc_to_chw(src, tmp);
        else
        {
          // LOG(INFO) << "tmp = src.getMat();";
          tmp = src.getMat();
        }

        if (tmp.isContinuous())
          mat = tmp;
        else
          tmp.copyTo(mat);
      }
      // LOG(INFO) << "finish mat:" << mat.total();

      // 2. 提取输入src， Tensor的维度
      TensorSize srcshape, validshape, alignedshape;
      shape(mat, srcshape);
      int cvmatByteSize = mat.total() * mat.elemSize();
      if (srcshape.dims() == 3) // 增加一个Batchsize维度
        srcshape.insert(0, 1);

      shape(property, validshape, false), shape(property, alignedshape, true);
      // LOG(INFO) << "srcshape: " << srcshape << ", validshape: " << validshape << ", alignedshape: " << alignedshape;
      int alignedByteSize = property.alignedByteSize;

      if (cvmatByteSize == alignedByteSize) // 3 如果字节数一样，检查下对齐shape是否匹配，如果匹配则直接赋值，否则报错。
      {
        if (srcshape == alignedshape)
        {
          auto data = dst.sysMem[0].virAddr;
          memcpy(reinterpret_cast<uint8_t *>(data), mat.data, cvmatByteSize);
        }
        else
        {
          std::stringstream ss;
          ss << "Invalid shape, input cv array shape: " << srcshape << ", aligned shape: " << alignedshape;
          CV_Error(cv::Error::StsAssert, ss.str());
        }
      }
      else if (cvmatByteSize < alignedByteSize) // 4. 若字节数不一样，检查下是否与valid匹配，匹配则做好对齐赋值，否则报错
      {
        if (srcshape == alignedshape)
        {
          auto data = dst.sysMem[0].virAddr;
          memcpy(reinterpret_cast<uint8_t *>(data), mat.data, cvmatByteSize);
        }
        else if (srcshape == validshape)
        {
          // LOG(INFO) << "aligned needed";
          alignMemory(mat.data, srcshape, (unsigned char *)dst.sysMem[0].virAddr, alignedshape, 1);
          // LOG(INFO) << "finish alignMemory";
        }
        else
        {
          std::stringstream ss;
          ss << "Invalid shape, input cv array shape: " << srcshape << ", valid shape: " << validshape;
          CV_Error(cv::Error::StsAssert, ss.str());
        }
      }
      else // 5. 待拷贝的字节数不可能大于对齐后的字节，输出报错信息
      {
        std::stringstream ss;
        ss << "ByteSizes are not equal. cvmatByteSize: " << cvmatByteSize << ", alignedByteSize: " << alignedByteSize;
        ss << "cv array shape: " << srcshape << ", valid shape: " << validshape << ", aligned shape: " << alignedshape;
        CV_Error(cv::Error::StsAssert, ss.str());
      }

      // 6. 根据flush决定是否刷新内存
      if (flush)
        flushBPU(dst, true);
    }

    void bpuMemcpy(hbDNNTensor &src, cv::OutputArray dst, bool align, bool flush)
    {
      HB_CHECK_SUCCESS(hbSysFlushMem(&src.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
                       "hbSysFlushMem tensor->cpu failed");

      // 1. 提取valid和align的shape
      auto &property = src.properties;
      TensorSize validshape, alignedshape;
      shape(property, validshape, false), shape(property, alignedshape, true);

      // 2. 判定类别
      bool fullsize = false, ishwc = false;
      std::vector<int> dims;
      if (align || validshape == alignedshape)
        alignedshape.copyTo(dims), fullsize = true;
      else
        validshape.copyTo(dims), fullsize = false;
      if (dims.size() == 4 && dims[0] == 1 && property.tensorLayout == HB_DNN_LAYOUT_NHWC) // HWC模式按照Opencv正常Mat形式存储
        ishwc = true;

      // 3. 分配内存
      switch (property.tensorType)
      {
      case HB_DNN_TENSOR_TYPE_S8:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_8S, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_8S, 1));
        break;
      case HB_DNN_TENSOR_TYPE_U8:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_8U, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_8U, 1));
        break;
      case HB_DNN_TENSOR_TYPE_S16:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_16S, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_16S, 1));
        break;
      case HB_DNN_TENSOR_TYPE_U16:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_16U, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_16U, 1));
        break;
      case HB_DNN_TENSOR_TYPE_F32:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_32F, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_32F, 1));
        break;
      case HB_DNN_TENSOR_TYPE_S32:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_32S, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_32S, 1));
        break;
      case HB_DNN_TENSOR_TYPE_F64:
        if (ishwc)
          dst.create(dims[1], dims[2], CV_MAKETYPE(CV_64F, dims[3]));
        else
          dst.create(dims.size(), &dims[0], CV_MAKETYPE(CV_64F, 1));
        break;
      default:
        CV_Error(cv::Error::StsAssert, "Unsupport type: " + formathbDNNDataType(hbDNNDataType(property.tensorType)));
        break;
      }

      // 4. 内存拷贝，如果fullsize为true，则不存在对齐问题，可以直接拷贝
      cv::Mat tmp = dst.getMat();
      if (fullsize)
      {
        int dstmemsize = tmp.total() * tmp.elemSize();
        // LOG(INFO) << "fullsize: " << dstmemsize;
        bpuMemcpy(tmp.data, src, dstmemsize, false);
      }
      else
      {
        alignMemory((unsigned char *)src.sysMem[0].virAddr, alignedshape, tmp.data, validshape, 1);
      }
    }

    void releaseTensors(std::vector<hbDNNTensor> &tensors)
    {
      for (int i = 0; i < tensors.size(); i++)
      {
        HB_CHECK_SUCCESS(hbSysFreeMem(&(tensors[i].sysMem[0])),
                         "hbDNNRelease tensor->cpu failed");
      }
    }

    void forward(const hbDNNHandle_t dnn_handle, const hbDNNTensor *_inTensors, hbDNNTensor *_outTensors, int waiting_time)
    {
      hbDNNInferCtrlParam infer_ctrl_param;
      hbDNNTaskHandle_t task_handle = nullptr;
      HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

      HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &_outTensors, _inTensors,
                                  dnn_handle, &infer_ctrl_param),
                       "hbDNNInfer failed");

      // wait task done
      HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, waiting_time), "hbDNNWaitTaskDone failed");

      // release task handle
      HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
    }

    void forward(const hbDNNHandle_t dnn_handle, const std::vector<hbDNNTensor> &inTensors, std::vector<hbDNNTensor> &outTensors, int waiting_time)
    {
      forward(dnn_handle, inTensors.data(), outTensors.data(), waiting_time);
    }

  } // end BPU
} // end wdr
