#ifndef WDR_TRACKER_H_
#define WDR_TRACKER_H_

#include <Core/core.h>
#include <BPU/bpu.h>

namespace wdr
{
  // common
  // 利用目标框估计方形边长
  float calTargetLength(const cv::Size2f &target, float thre_context = 0.5);
  float calTargetLength(float tw, float th, float thre_context = 0.5);

  void get_subwindow_tracking(const cv::Mat &src, cv::Mat &dst, const cv::Rect2f &croproi, bool usecenter, const cv::Size &modelsize);

  // lsrc估计出的边长，ldst目标边长，target实际的目标框
  cv::Rect2f estRectInCrop(float lsrc, float ldst, const cv::Size2f &target);

  cv::Rect2f recoverNewBox(const cv::Size2f &crop, const cv::Rect2f &pred,
                           const cv::Rect2f &src,
                           const cv::Point2f &scalexy, float lr);
  cv::Rect normBox(const cv::Rect2f &box, const cv::Size &size);

  cv::Rect2f estimateTrackRect(const cv::Mat &anchorx, const cv::Mat &anchory,
                               const cv::Mat &window,
                               const cv::Rect2f &sz_in,
                               const cv::Mat &scores, const cv::Mat &preds,
                               const cv::Point2f &scalexy,
                               float penalty_k = 0.007,
                               float window_influence = 0.225,
                               float lr = 0.616);

  // 构造anchor
  void grids(const cv::Size &size, cv::Mat &gridx, cv::Mat &gridy);

  // DCMT的代码参考: https://github.com/Z-Xiong/LightTrack-ncnn
  struct DCMTConfigs
  {
    int total_stride{16};
    int instance_size{288};
    float context_amount{0.5f};
    int exemplar_size{127};
    float penalty_tk{0.007};
    float window_influence{0.225};
    float lr{0.616};
  };

  class TrackerDCMT
  {
  public:
    TrackerDCMT(const DCMTConfigs &config, const std::string &modelpath, const std::string &modelname);

    // 初始化相关
    void initBPU(const std::string &modelpath, const std::string &modelname);
    void reset(); ////// 通用变量
    // 跟踪相关
    void init(const cv::Mat &im, const cv::Rect2f &target);
    void track(const cv::Mat &img);

    void update(const cv::Mat &x_crops, float scale_z);

    static void get_subwindow_tracking(const cv::Mat &src, cv::Mat &dst, const cv::Point2f pos, int model_sz, int original_sz);
    static void get_bbox(int s_z, int exemplar_size, const cv::Size2f &target_sz, cv::Rect2f &bbox);

  private:
    DCMTConfigs config;

    // BPU推理变量
    BPU::BpuNets nets;
    int idxnet{0};
    BPU::BpuMats input_mats, output_mats;
    cv::Size outsize;

    // 初始化变量, 在reset中更新
    cv::Mat anchorx, anchory;
    float d_search{0};
    cv::Mat window;

    // 跟踪相关
    void set_target(const cv::Rect2f &target, bool center = false); // 跟踪框信息
    cv::Rect2f get_target() const;
    void norm_target(int rows, int cols);
    cv::Point2f target_pos{0.0, 0.0}; // 跟踪框中心点
    cv::Size2f target_size{0.0, 0.0}; // 跟踪框尺寸

  };
}

#endif