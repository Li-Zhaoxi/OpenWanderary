
#include <BPU/bpu.h>
#include <boost/filesystem.hpp>
#include <unistd.h>

namespace wdr
{

  namespace BPU
  {

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

    void BpuNets::init(int idx, BpuMats &input_mats, BpuMats &output_mats, bool autopadding) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + " is valid, total number is " + std::to_string(this->total()));

      input_mats.create(netinfos[idx].input_infos, autopadding);
      output_mats.create(netinfos[idx].output_infos, false);
    }

    void BpuNets::forward(int idx, const BpuMats &input_mats, BpuMats &output_mats) const
    {
      if (input_mats.device(-1) == DEVICE::NET_CPU_BPU)
        CV_Error(cv::Error::StsError, "Input Tensors are not all in BPU, Please call input_mats.bpu() before forward().");

      // if (output_mats.device() != DEVICE::NET_BPU)
      //   CV_Error(cv::Error::StsError, "Output Tensors are not in BPU, Please call input_mats.bpu() before forward().");

      // LOG(INFO) << "net debug 1";
      std::string errmsg;
      if (!checkTensorProperties(idx, input_mats, true, errmsg))
        CV_Error(cv::Error::StsOutOfRange, errmsg);
      // LOG(INFO) << "net debug 2";
      if (!checkTensorProperties(idx, output_mats, false, errmsg))
        CV_Error(cv::Error::StsOutOfRange, errmsg);
      // LOG(INFO) << "net debug 3";
      const hbDNNTensor *_inputtensor = input_mats.matset->data() + input_mats.range.start;
      hbDNNTensor *_outputtensor = output_mats.matset->data() + output_mats.range.start;
      // LOG(INFO) << "in tensor 0: " << _inputtensor[0].properties;
      // LOG(INFO) << "in tensor 1: " << _inputtensor[1].properties;
      // LOG(INFO) << "in tensor 2: " << _inputtensor[2].properties;
      // LOG(INFO) << "out tensor 0: " << _outputtensor[0].properties;
      // LOG(INFO) << "out tensor 1: " << _outputtensor[1].properties;
      // LOG(INFO) << "in size: " << input_mats.matset->size() << ", start: " << input_mats.range.start;
      // LOG(INFO) << "out size: " << output_mats.matset->size() << ", start: " << output_mats.range.start;
      // LOG(INFO) << "net debug 4";
      // LOG(INFO) << "idx: " << idx;
      // LOG(INFO) << "handle: " << netsMap[idx].second << ", name: " << netsMap[idx].first;
      wdr::BPU::forward(netsMap[idx].second, _inputtensor, _outputtensor);
      // LOG(INFO) << "net debug 5";
      output_mats.end_forwart();
      // LOG(INFO) << "net debug 6";
    }

  } // end BPU
} // end wdr