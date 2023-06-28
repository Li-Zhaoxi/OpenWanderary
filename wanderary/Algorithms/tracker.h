#ifndef WDR_TRACKER_H_
#define WDR_TRACKER_H_

#include <Core/core.h>
#include <BPU/bpu.h>

namespace wdr
{

  struct DCMTConfigs
  {
    struct
    {
      std::string LOG_DIR{"logs"};
      std::string CHECKPOINT_DIR{"re_snapshot"};
      bool USE_CUDA{true};
      std::string GPUS{"0"};
    } COMMON;

    struct
    {
      std::string NAME{"DenseSiam"};
      int STRIDE{8};
      struct
      {
        std::string NAME{"ResNet50Dilated"};
        std::vector<int> LAYER{{3}};
      } BACKBONE;
    } MODEL;
  };

  class TrackerDCMT
  {
  public:
    TrackerDCMT(const DCMTConfigs &config, const std::string &modelpath, const std::string &modelname);
    void init(const cv::Mat &im, const cv::Point &target_pos, const cv::Size &target_sz);

  private:
    DCMTConfigs config;
    int stride;

    // BPU推理变量
    BPU::BpuNets nets;
    int idxnet{0};
    BPU::BpuMats input_mats, output_mats;
  };
}

#endif