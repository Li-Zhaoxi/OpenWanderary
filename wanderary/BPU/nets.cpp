
#include <BPU/bpu.h>
#include <boost/filesystem.hpp>
#include <unistd.h>

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

    ////////////////////// BpuNets /////////////////
    BpuNets::BpuNets()
    {
      if (getuid())
        CV_Error(cv::Error::StsError, "You must use ROOT or SUDO to use these BPU functions.");
    }
    void BpuNets::release()
    {
      if (pPackedNets)
      {
        releaseNets(pPackedNets);
        pPackedNets = nullptr;
        netsMap.clear(), netinfos.clear();
      }
    }

    BpuNets::~BpuNets()
    {
      release();
    }

    void BpuNets::readNets(const std::vector<std::string> &modelpaths)
    {
      if (pPackedNets)
        release();

      for (const auto &modelpath : modelpaths)
      {
        if (!boost::filesystem::is_regular_file(boost::filesystem::path(modelpath)))
          CV_Error(cv::Error::StsAssert, "The file does not exist: " + modelpath);
      }

      std::unordered_map<std::string, hbDNNHandle_t> netstmp;
      wdr::BPU::readNets(modelpaths, pPackedNets, netstmp);

      for (const auto &net : netstmp)
      {
        netsMap.push_back(std::make_pair(net.first, net.second));

        netinfos.emplace_back();
        auto &info = netinfos.back();
        info.modelname = net.first;
        readNetProperties(net.second, info.input_infos.infos, true);
        readNetProperties(net.second, info.output_infos.infos, false);
      }
    }

    int BpuNets::name2index(const std::string &modelname) const
    {
      int idx = -1;
      const int modelnum = this->total();
      for (int k = 0; k < modelnum; k++)
      {
        if (netsMap[k].first == modelname)
        {
          idx = k;
          break;
        }
      }

      return idx;
    }

    const std::string &BpuNets::index2name(int idx) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + " is valid, total number is " + std::to_string(this->total()));

      return netsMap[idx].first;
    }

    bool BpuNets::checkTensorProperties(int idx, const BpuMats &bpumats, bool input, std::string &errmsg) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + " is valid, total number is " + std::to_string(this->total()));

      const auto &dstnetinfos = netinfos[idx];
      const auto &matinfos = input ? dstnetinfos.input_infos : dstnetinfos.output_infos;

      errmsg = "";
      std::stringstream ss;

      // 检查个数一致性
      const int srcnum = bpumats.size(), dstnum = matinfos.size();
      if (srcnum != dstnum)
      {
        ss << "Tensor numbers are not equal. BpuMats num: " << srcnum << ", real num: " << dstnum;
        errmsg = ss.str();
        return false;
      }

      for (int k = 0, r = bpumats.range.start; k < srcnum, r < bpumats.range.end; k++, r++)
      {
        const auto &srcinfo = bpumats.properties->infos[r];
        const auto &dstinfo = matinfos.infos[k];

        // 检查Tensor类型
        if (srcinfo.tensorType != dstinfo.tensorType)
        {
          ss << k << "th tensor dtypes are not equal. Expected type is " << formathbDNNQuantiType(hbDNNQuantiType(srcinfo.tensorType))
             << ", but the used one is " << formathbDNNQuantiType(hbDNNQuantiType(dstinfo.tensorType));
          errmsg = ss.str();
          return false;
        }

        // 检查Tensor字节数
        if (srcinfo.alignedByteSize != dstinfo.alignedByteSize)
        {
          ss << k << "th tensor alignedByteSize are not equal. Expected size is " << srcinfo.alignedByteSize
             << ", but the used one is " << dstinfo.alignedByteSize;
          errmsg = ss.str();
          return false;
        }

        // 检查未对齐的维度
        const auto &srcshape = srcinfo.validShape;
        const auto &dstshape = dstinfo.validShape;
        if (srcshape.numDimensions != dstshape.numDimensions)
        {
          ss << k << "th tensor validShape numDimensions are not equal. Expected dim is " << srcshape.numDimensions
             << ", but the used one is " << dstshape.numDimensions;
          errmsg = ss.str();
          return false;
        }
        for (int idxdim = 0; idxdim < srcshape.numDimensions; idxdim++)
        {
          if (srcshape.dimensionSize[idxdim] != dstshape.dimensionSize[idxdim])
          {
            ss << idxdim << "th dims of " << k << "th tensor validShape are not equal. Expected dim is " << srcshape.dimensionSize[idxdim]
               << ", but the used one is " << dstshape.dimensionSize[idxdim];
            errmsg = ss.str();
            return false;
          }
        }
      }

      return true;
    }

    void BpuNets::init(int idx, BpuMats &input_mats, BpuMats &output_mats) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + " is valid, total number is " + std::to_string(this->total()));

      input_mats.create(netinfos[idx].input_infos);
      output_mats.create(netinfos[idx].output_infos);
    }

    void BpuNets::forward(int idx, const BpuMats &input_mats, BpuMats &output_mats) const
    {
      if (input_mats.device() != DEVICE::NET_BPU)
        CV_Error(cv::Error::StsError, "Input Tensors are not in BPU, Please call input_mats.bpu() before forward().");

      if (output_mats.device() != DEVICE::NET_BPU)
        CV_Error(cv::Error::StsError, "Output Tensors are not in BPU, Please call input_mats.bpu() before forward().");

      std::string errmsg;
      if (!checkTensorProperties(idx, input_mats, true, errmsg))
        CV_Error(cv::Error::StsOutOfRange, errmsg);

      if (!checkTensorProperties(idx, output_mats, false, errmsg))
        CV_Error(cv::Error::StsOutOfRange, errmsg);

      const hbDNNTensor *_inputtensor = input_mats.matset->data() + input_mats.range.start;
      hbDNNTensor *_outputtensor = output_mats.matset->data() + output_mats.range.start;
      wdr::BPU::forward(netsMap[idx].second, _inputtensor, _outputtensor);
    }

  } // end BPU
} // end wdr