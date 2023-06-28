#include <Algorithms/tracker.h>

namespace wdr
{
  TrackerDCMT::TrackerDCMT(const DCMTConfigs &config, const std::string &modelpath, const std::string &modelname)
  {
    this->config = config;
    this->stride = config.MODEL.STRIDE;

    // 加载BPU网络并初始化参数
    nets.readNets({modelpath});
    int idxnet = nets.name2index(modelname);
    std::string modelname = nets.index2name(idxnet);
    LOG(INFO) << "DCMT [" << modelname << "] at " << idxnet << " has been loaded.";
    nets.init(idxnet, input_mats, output_mats, true);
    LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();
  }

  void TrackerDCMT::init(const cv::Mat &im, const cv::Point &target_pos, const cv::Size &target_sz)
  {
    
  }
}