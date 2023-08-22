#include <Algorithms/tracker.h>

namespace wdr
{

  // void TrackerDCMT::grids(int score_size, int total_stride, std::vector<cv::Point2f> &grid_to_search)
  // {
  //   const int sz = score_size, total = score_size * score_size; // 18

  //   grid_to_search.resize(total);
  //   for (int i = 0; i < sz; i++)
  //   {
  //     int idxi = i * sz;
  //     for (int j = 0; j < sz; j++)
  //     {
  //       grid_to_search[idxi + j].x = j * total_stride;
  //       grid_to_search[idxi + j].y = i * total_stride;
  //     }
  //   }
  // }

  void TrackerDCMT::get_subwindow_tracking(const cv::Mat &src, cv::Mat &dst, const cv::Point2f pos, int model_sz, int original_sz)
  {
    float c = (original_sz + 1.0f) / 2.0f;

    int context_xmin, context_xmax, context_ymin, context_ymax;
    context_xmin = int(pos.x - c + 0.5), context_xmax = context_xmin + original_sz - 1;
    context_ymin = int(pos.y - c + 0.5), context_ymax = context_ymin + original_sz - 1;

    int left_pad, top_pad, right_pad, bottom_pad;
    left_pad = std::max(0, -context_xmin), top_pad = std::max(0, -context_ymin);
    right_pad = std::max(0, context_xmax - src.cols + 1);
    bottom_pad = std::max(0, context_ymax - src.rows + 1);

    context_xmin += left_pad, context_xmax += left_pad;
    context_ymin += top_pad, context_ymax += top_pad;

    cv::Rect roi(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1);

    cv::Mat im_path_original, te_im;
    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
      cv::copyMakeBorder(src, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT);
      im_path_original = te_im(roi);
    }
    else
      im_path_original = src(roi);

    cv::resize(im_path_original, dst, cv::Size(model_sz, model_sz));
  }

  TrackerDCMT::TrackerDCMT(const DCMTConfigs &config, const std::string &modelpath, const std::string &modelname)
  {
    this->config = config;

    this->initBPU(modelpath, modelname);
    this->reset();
  }

  void TrackerDCMT::initBPU(const std::string &modelpath, const std::string &modelname)
  {
    nets.readNets({modelpath});
    idxnet = nets.name2index(modelname);
    LOG(INFO) << "DCMT [" << modelname << "] at " << idxnet << " has been loaded.";

    nets.init(idxnet, input_mats, output_mats, true);
    LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();

    outsize = output_mats[0].size();
  }

  void TrackerDCMT::reset()
  {
    // 初始化Anchor
    wdr::grids(outsize, anchorx, anchory);
    anchorx = anchorx * config.total_stride + config.instance_size / 2;
    anchory = anchory * config.total_stride + config.instance_size / 2;

    // 更新dsearch: backbone_model_size - init_model_size = 288-127
    d_search = (float(config.instance_size) - config.exemplar_size) / 2;

    // 初始化window
    cv::Mat hr, hc;
    wdr::hanning(outsize.height, hr, CV_32F);
    wdr::hanning(outsize.width, hc, CV_32F);
    window = hr * hc.t();
  }

  void TrackerDCMT::set_target(const cv::Rect2f &target, bool center)
  {
    if (center)
      this->target_pos.x = target.x, this->target_pos.y = target.y;
    else
    {
      this->target_pos.x = target.x + target.width / 2.0f;
      this->target_pos.y = target.y + target.height / 2.0f;
    }

    this->target_size.width = target.width;
    this->target_size.height = target.height;
  }

  cv::Rect2f TrackerDCMT::get_target() const
  {
    cv::Rect2f res;
    res.x = target_pos.x - target_size.width / 2;
    res.y = target_pos.y - target_size.height / 2;

    res.width = target_size.width, res.height = target_size.height;
    return res;
  }

  void TrackerDCMT::init(const cv::Mat &im, const cv::Rect2f &target)
  {
    set_target(target, false);

    // 构建Crop
    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。
    // 之后将其resize为127x127x3.成为模板图像
    cv::Mat z_bgr;
    float s_zf;
    {
      cv::Rect2f tgtf(target.x, target.y, target.width, target.height);
      s_zf = wdr::calTargetLength(tgtf.size(), config.context_amount); // 计算模板边长
      // 从原图中提取网络输入的ROI，并resize
      wdr::get_subwindow_tracking(im, z_bgr,
                                  cv::Rect2f(target_pos.x, target_pos.y, s_zf, s_zf), true,
                                  cv::Size(config.exemplar_size, config.exemplar_size));
    }

    // 构建z_box
    cv::Mat z_box(4, 1, CV_32FC1);
    {
      cv::Rect2f _zbbox = wdr::estRectInCrop(s_zf, config.exemplar_size, this->target_size);
      z_box.at<float>(0) = _zbbox.x, z_box.at<float>(1) = _zbbox.y;
      z_box.at<float>(2) = _zbbox.x + _zbbox.width, z_box.at<float>(3) = _zbbox.y + _zbbox.height;
    }

    // 上传z_bgr和z_box到BPU
    wdr::BPU::BpuMat bpu_z_bgr = input_mats[1];
    bpu_z_bgr << z_bgr, bpu_z_bgr.bpu();

    wdr::BPU::BpuMat bpu_z_box = input_mats[2];
    bpu_z_box << z_box, bpu_z_box.bpu();
  }

  void TrackerDCMT::track(const cv::Mat &img)
  {
    cv::Mat x_crop;
    float scale_z;
    {
      // 利用当前target估计一个边长
      float s_z = calTargetLength(target_size, config.context_amount);

      // 利用边长大小估计实际search大小
      // 由于已经是方形了，因此主要是估计新的方形边长 s_x
      scale_z = config.exemplar_size / s_z; // 127/
      float pad = d_search / scale_z;
      float s_x = s_z + 2 * pad;

      wdr::get_subwindow_tracking(img, x_crop,
                                  cv::Rect2f(target_pos.x, target_pos.y, s_x, s_x), true,
                                  cv::Size(config.instance_size, config.instance_size));
    }

    // 上传数据并进行推理
    cv::Mat _cls_score, _bbox_pred;
    {
      wdr::BPU::BpuMat bpu_crop = input_mats[0];
      bpu_crop << x_crop, bpu_crop.bpu();
      nets.forward(idxnet, input_mats, output_mats);

      wdr::BPU::BpuMat _score = output_mats[0], _predbbox = output_mats[1];
      _score.cpu(), _score >> _cls_score;
      _predbbox.cpu(), _predbbox >> _bbox_pred;
    }

    cv::Rect2f res = wdr::estimateTrackRect(anchorx, anchory, window, 
                                            get_target(), 
                                            _cls_score, _bbox_pred,
                                            cv::Point2f(scale_z, scale_z),
                                            config.penalty_tk,
                                            config.window_influence,
                                            config.lr);
    set_target(res, true);
    norm_target(img.rows, img.cols);
  }

  void TrackerDCMT::norm_target(int rows, int cols)
  {
    target_pos.x = std::max(0.0f, std::min(cols * 1.0f, target_pos.x));
    target_pos.y = std::max(0.0f, std::min(rows * 1.0f, target_pos.y));
    target_size.width = std::max(10.0f, std::min(cols * 1.0f, target_size.width));
    target_size.height = std::max(10.0f, std::min(rows * 1.0f, target_size.width));
  }
}