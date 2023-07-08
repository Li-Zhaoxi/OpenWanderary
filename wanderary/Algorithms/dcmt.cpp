#include <Algorithms/tracker.h>

namespace wdr
{

  void TrackerDCMT::get_bbox(int s_z, int exemplar_size, const cv::Size2f &target_sz, cv::Rect2f &bbox)
  {
    float scale_z, w, h;
    // map the GT bounding box in the first frame to template (127*127)
    scale_z = float(exemplar_size) / s_z;
    w = target_sz.width * scale_z, h = target_sz.height * scale_z;

    int cx = exemplar_size / 2, cy = exemplar_size / 2;

    bbox.x = cx - w * 0.5f;
    bbox.y = cy - h * 0.5f;
    bbox.width = w, bbox.height = h;
  }

  void TrackerDCMT::grids(int score_size, int total_stride, cv::Mat &grid_to_search)
  {
    const int sz = score_size, total = score_size * score_size; // 18

    grid_to_search.create(total, 2, CV_32FC1);
    float *_grid_to_search = (float *)grid_to_search.data;
    for (int i = 0; i < sz; i++)
    {
      int idxi = i * sz;
      for (int j = 0; j < sz; j++)
      {
        int idx = (idxi + j) * 2;
        _grid_to_search[idx] = j * total_stride;
        _grid_to_search[idx + 1] = i * total_stride;
      }
    }
  }
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
    score_size = int(config.instance_size * 1.0f / config.total_stride + 0.5f);

    // 加载BPU网络并初始化参数
    nets.readNets({modelpath});
    int idxnet = nets.name2index(modelname);
    std::string modelname = nets.index2name(idxnet);
    LOG(INFO) << "DCMT [" << modelname << "] at " << idxnet << " has been loaded.";
    nets.init(idxnet, input_mats, output_mats, true);
    LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();

    // 变量初始化
    shape_cls = output_mats[0].size();
    // backbone_model_size - init_model_size = 288-127
    float d_search = (config.instance_size - config.exemplar_size) / 2;

    grids(score_size, config.total_stride, grid_to_search);
  }

  void TrackerDCMT::init(const cv::Mat &im, const cv::Rect &target)
  {
    this->ori_img = im.size();
    this->target_sz.x = target.width, this->target_sz.y = target.height;
    this->target_pos.x = target.x + target.width / 2.0f;
    this->target_pos.y = target.y + target.height / 2.0f;

    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。之后将其resize为127x127x3.成为模板图像
    // context = 1/2 * (w+h) = 2*pad
    float wc_z = target_sz.x + config.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + config.context_amount * (target_sz.x + target_sz.y);

    // 计算sz_wh
    float pad = (target_sz.x + target_sz.y) * 0.5f;
    sz_wh = std::sqrt((target_sz.x + pad) * (target_sz.y + pad));

    // z_crop size = sqrt((w+2p)*(h+2p))
    int s_z = int(sqrt(wc_z * hc_z) + 0.5); // orignal size

    cv::Mat z_crop;
    get_subwindow_tracking(im, z_crop, target_pos, config.exemplar_size, s_z);
    if (z_crop.isContinuous())
      z_bgr = z_crop;
    else
      z_crop.copyTo(z_bgr);

    cv::Rect2f _zbbox;
    get_bbox(s_z, config.exemplar_size, cv::Size2f(wc_z, hc_z), _zbbox);
    z_box.create(4, 1, CV_32FC1);
    z_box.at<float>(0) = _zbbox.x, z_box.at<float>(1) = _zbbox.y;
    z_box.at<float>(2) = _zbbox.x + _zbbox.width, z_box.at<float>(3) = _zbbox.y + _zbbox.height;

    cv::Mat hanning = cv::Mat::zeros(score_size, 1, CV_32FC1);
    float *_hanning = (float *)hanning.data;
    for (int i = 0; i < score_size; i++)
      _hanning[i] = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
    window = hanning * hanning.t();

    // 上传z_bgr和z_box到BPU
    input_mats[1] << z_bgr, input_mats[2] << z_box;
  }

  void TrackerDCMT::update(const cv::Mat &x_crops, float scale_z)
  {
    cv::Mat _cls_score, _bbox_pred;
    input_mats[0] << x_crops;
    nets.forward(idxnet, input_mats, output_mats);
    output_mats.cpu();
    output_mats[0] >> _cls_score, output_mats[1] >> _bbox_pred;

    CV_Assert(_cls_score.depth() == CV_32F && _bbox_pred.depth() == CV_32F);
    // CV_Assert(_bbox_pred)

    // _cls_score维度：[1x1x31x31], _bbox_pred维度：[1x4x31x31]
    cv::Mat cls_score, bbox_pred;
    cv::Mat(shape_cls.height * shape_cls.width, 1, CV_32FC1, _cls_score.data).copyTo(cls_score);
    wdr::sigmode((float *)cls_score.data, cls_score.total()); // 31x31

    cv::Mat pred1 = grid_to_search - cv::Mat(2, cls_score.total(), CV_32FC1, _bbox_pred.data).t();
    cv::Mat pred2 = grid_to_search - cv::Mat(2, cls_score.total(), CV_32FC1, _bbox_pred.data + cls_score.total() * _bbox_pred.elemSize1()).t(); // 注意elemSize1和注意elemSize的区别，前面的一些代码要修改

    cv::Mat wh = pred2 - pred1, s_c, r_c;
    cv::Mat pad = (wh.col(0) + wh.col(1)) / 2.0f;
    cv::sqrt(((wh.col(0) + pad) * (wh.col(1) + pad)) / sz_wh, s_c);

    float ratio = target_sz.x / target_sz.y;
    cv::Mat t = ratio / wh.col(0) / wh.col(1);
    cv::max(t, 1.0f / t, r_c);

    cv::Mat penalty;
    cv::exp((1 - s_c.mul(r_c)) * config.penalty_tk, penalty);

    cv::Mat pscore = penalty.mul(cls_score) * (1 - config.window_influence) + window * config.window_influence;
    double maxScore = 0;
    int idxmaxl;
    cv::minMaxIdx(pscore, nullptr, &maxScore, nullptr, &idxmaxl);
    int r_max, c_max;
    r_max = idxmaxl / shape_cls.width, c_max = idxmaxl % shape_cls.width; // 这里存疑，需要验证

    float pred_x1_real, pred_y1_real, pred_x2_real, pred_y2_real;
    pred_x1_real = pred1.at<float>(idxmaxl, 0);
    pred_y1_real = pred1.at<float>(idxmaxl, 1);
    pred_x2_real = pred2.at<float>(idxmaxl, 0);
    pred_y2_real = pred2.at<float>(idxmaxl, 1);

    float pred_xs, pred_ys, pred_w, pred_h, diff_xs, diff_ys;
    pred_xs = (pred_x1_real + pred_x2_real) / 2, pred_ys = (pred_y1_real + pred_y2_real) / 2;
    pred_w = pred_x2_real - pred_x1_real, pred_h = pred_y2_real - pred_y1_real;
    diff_xs = pred_xs - config.instance_size / 2, diff_ys = pred_ys - config.instance_size / 2;

    diff_xs /= scale_z, diff_ys /= scale_z, pred_w /= scale_z, pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z, target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr_ = penalty.at<float>(idxmaxl) * cls_score.at<float>(idxmaxl) * config.lr;
    // size rate
    float res_xs, res_ys, res_w, res_h;
    res_xs = target_pos.x + diff_xs, res_ys = target_pos.y + diff_ys;
    res_w = pred_w * config.lr + (1 - lr_) * target_sz.x;
    res_h = pred_h * config.lr + (1 - lr_) * target_sz.y;

    // 这里存疑，看看优化
    target_pos_int.x = int(res_xs), target_pos_int.y = int(res_ys);
    target_sz.x = target_sz.x * (1 - lr_) + lr_ * res_w;
    target_sz.y = target_sz.y * (1 - lr_) + lr_ * res_h;
  }

  void TrackerDCMT::track(const cv::Mat &img)
  {
    float hc_z, wc_z, s_z, scale_z;
    hc_z = target_sz.y + config.context_amount * (target_sz.x + target_sz.y);
    wc_z = target_sz.x + config.context_amount * (target_sz.x + target_sz.y);
    s_z = sqrt(wc_z * hc_z);              // roi size
    scale_z = config.exemplar_size / s_z; // 127/

    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    cv::Mat x_crop;
    get_subwindow_tracking(img, x_crop, target_pos, config.instance_size, int(s_x));

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    this->update(x_crop, scale_z);
    target_pos.x = std::max(0.0f, std::min(img.cols * 1.0f, target_pos.x));
    target_pos.y = std::max(0.0f, std::min(img.rows * 1.0f, target_pos.y));
    target_sz.x = float(std::max(10, std::min(img.cols, int(target_sz.x))));
    target_sz.y = float(std::max(10, std::min(img.rows, int(target_sz.y))));
  }
}