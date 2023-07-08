#ifndef WDR_TRACKER_H_
#define WDR_TRACKER_H_

#include <Core/core.h>
#include <BPU/bpu.h>

namespace wdr
{

  // DCMT的代码参考: https://github.com/Z-Xiong/LightTrack-ncnn
  struct DCMTConfigs
  {
    int total_stride{16};
    int instance_size{288};
    float context_amount{0.5f};
    int exemplar_size{127};
    float penalty_tk{0.007};
    float window_influence{0.225};
    int instance_size{288};
    float lr{0.616};
  };

  class TrackerDCMT
  {
  public:
    TrackerDCMT(const DCMTConfigs &config, const std::string &modelpath, const std::string &modelname);
    void init(const cv::Mat &im, const cv::Rect &target);
    void update(const cv::Mat &x_crops, float scale_z);

    static void get_subwindow_tracking(const cv::Mat &src, cv::Mat &dst, const cv::Point2f pos, int model_sz, int original_sz);
    static void grids(int score_size, int total_stride, cv::Mat &grid_to_search);
    static void get_bbox(int s_z, int exemplar_size, const cv::Size2f &target_sz, cv::Rect2f &bbox);

  private:
    DCMTConfigs config;

    //
    int score_size{1};
    cv::Size ori_img, shape_cls;
    cv::Point2f target_pos{0.0, 0.0}, target_sz{0.0, 0.0};
    cv::Point target_pos_int;
    cv::Mat grid_to_search, window, z_bgr, z_box;
    float sz_wh{1};

    // BPU推理变量
    BPU::BpuNets nets;
    int idxnet{0};
    BPU::BpuMats input_mats, output_mats;
  };
}

#endif