#include <Algorithms/tracker.h>

namespace wdr
{
  float calTargetLength(const cv::Size2f &target, float thre_context)
  {
    // context = 1/2 * (w+h) = 2*pad
    const float lwh = (target.height + target.width) * thre_context;
    float wc_z = target.width + lwh, hc_z = target.height + lwh;
    return std::sqrt(wc_z * hc_z);
  }

  float calTargetLength(float tw, float th, float thre_context)
  {
    return calTargetLength(cv::Size2f(tw, th), thre_context);
  }

  void get_subwindow_tracking(const cv::Mat &src, cv::Mat &dst, const cv::Rect2f &croproi, bool usecenter, const cv::Size &modelsize)
  {
    // 构建ROI的xy的最大值和最小值
    int context_xmin{0}, context_xmax{0}, context_ymin{0}, context_ymax{0};
    if (usecenter)
    {
      float cw = (croproi.width + 1.0f) / 2.0f, ch = (croproi.height + 1.0f) / 2.0f;
      context_xmin = int(croproi.x - cw + 0.5), context_xmax = int(context_xmin + croproi.width + 0.5);
      context_ymin = int(croproi.y - ch + 0.5), context_ymax = int(context_ymin + croproi.height + 0.5);
    }
    else
    {
      context_xmin = int(croproi.x), context_xmax = int(context_xmin + croproi.width + 0.5);
      context_ymin = int(croproi.y), context_ymax = int(context_ymin + croproi.height + 0.5);
    }

    // 计算ROI的越界情况
    int left_pad{0}, top_pad{0}, right_pad{0}, bottom_pad{0};
    left_pad = std::max(0, -context_xmin), top_pad = std::max(0, -context_ymin);
    right_pad = std::max(0, context_xmax - src.cols), bottom_pad = std::max(0, context_ymax - src.rows);

    cv::Mat imgroi;
    if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
    {
      cv::Rect roi(context_xmin, context_ymin, context_xmax - context_xmin, context_ymax - context_ymin);
      imgroi = src(roi);
    }
    else
    {
      cv::Rect srcroi(context_xmin + left_pad, context_ymin + top_pad, context_xmax - right_pad, context_ymax - bottom_pad);
      srcroi.width = srcroi.width - srcroi.x, srcroi.height = srcroi.height - srcroi.y;
      cv::Rect dstroi(left_pad, top_pad, srcroi.width, srcroi.height);

      // 初始化
      imgroi.create(modelsize.height, modelsize.width, CV_MAKETYPE(src.depth(), src.channels()));
      imgroi.setTo(0);
      src(srcroi).copyTo(imgroi(dstroi));
    }

    cv::resize(imgroi, dst, modelsize);
  }

  // lsrc估计出的边长，ldst目标边长，target实际的目标框
  cv::Rect2f estRectInCrop(float lsrc, float ldst, const cv::Size2f &target)
  {
    float scale = ldst / lsrc;
    float w = target.width * scale, h = target.height * scale;

    cv::Rect2f res;
    res.x = ldst / 2.0f - w / 2.0f;
    res.y = ldst / 2.0f - h / 2.0f;
    res.width = w, res.height = h;
    return res;
  }

  cv::Rect2f recoverNewBox(const cv::Size2f &crop, const cv::Rect2f &pred,
                         const cv::Rect2f &src,
                         const cv::Point2f &scalexy, float lr)
  {
    cv::Rect2f newbox;

    ////// 估计实际中心点
    // 计算估计的目标框相对于输入目标框的偏差
    float diff_xs = pred.x - crop.width / 2, diff_ys = pred.y - crop.height / 2;
    diff_xs /= scalexy.x, diff_ys /= scalexy.y;
    newbox.x = src.x + diff_xs, newbox.y = src.y + diff_ys;

    ////// 估计实际尺寸（加权）
    newbox.width = src.width * (1 - lr) + pred.width * lr;
    newbox.height = src.height * (1 - lr) + pred.height * lr;
    newbox.width /= scalexy.x, newbox.height /= scalexy.y;

    return newbox;
  }

  cv::Rect normBox(const cv::Rect2f &box, const cv::Size &size)
  {
    cv::Rect res;
    res.x = std::max(0, int(box.x + 0.5));
    res.y = std::max(0, int(box.y + 0.5));
    res.width = std::min(size.width, int(box.width + 0.5));
    res.height = std::min(size.height, int(box.height + 0.5));

    return res;
  }

  cv::Rect2f estimateTrackRect(const cv::Mat &anchorx, const cv::Mat &anchory,
                               const cv::Mat &window,
                               const cv::Rect2f &sz_in,
                               const cv::Mat &scores, const cv::Mat &preds,
                               const cv::Point2f &scalexy,
                               float penalty_k,
                               float window_influence,
                               float lr)
  {
    // Lambda 函数集
    auto fun_sz = [](float w, float h)
    {
      float pad = (w + h) * 0.5f;
      return std::sqrt((w + pad) * (h + pad));
    };

    auto fun_rnorm = [](float r)
    {
      return std::max(r, 1.0f / r);
    };

    // 确认维度，确认数据类型，确认内存连续
    const int rows = anchorx.rows, cols = anchorx.cols, total = rows * cols;

    // 预分配所有估计出的框
    std::vector<cv::Rect2f> bboxs(total);
    cv::Mat costs(1, total, CV_64FC1), pscore(1, total, CV_32FC1); // CV_64FC1 方便后续计算最大值用
    double *_costs = (double *)costs.data;
    float *_pscore = (float *)pscore.data;

    // 多线程解算目标框
    float ltar_diag = fun_sz(sz_in.width, sz_in.height);
    float rtar_wh = sz_in.width / sz_in.height;

    float *_anchorx = (float *)anchorx.data, *_anchory = (float *)anchory.data;
    float *_window = (float *)window.data;
    float *_scores = (float *)scores.data, *_preds = (float *)preds.data;

    auto fun_cal_bbox = [&](cv::Range range)
    {
      for (int r = range.start; r < range.end; r++)
      {
        int idxr = r / cols, idxc = r % cols;

        // 1. 估计左上角和右下角预测点
        cv::Point2f lt, br;
        {
          float anchorx = _anchorx[r], anchory = _anchory[r];
          lt.x = anchorx - _preds[r];
          lt.y = anchory - _preds[r + total];
          br.x = anchorx + _preds[r + total * 2];
          br.y = anchory + _preds[r + total * 3];
        }
        cv::Size wh(br.x - lt.x, br.y - lt.y);

        // 2. 估计预测评分
        // 2.1 尺寸/宽高比关联的损失值
        float cost_sr;
        {
          float score = wdr::sigmode(_scores[r]);
          // size penalty: 两个框带0.5pad的对角线长度比值
          float cost_size = fun_rnorm(fun_sz(wh.width, wh.height) / ltar_diag);
          // ratio penalty: 两个框的宽高比的比值
          float cost_ratio = fun_rnorm((wh.width / wh.height / rtar_wh));
          cost_sr = std::exp(-(cost_size * cost_ratio - 1) * penalty_k);
        }
        // 2.2 引入window损失，得到最终评分
        float score_final = cost_sr * (1 - window_influence) + _window[r] * window_influence;

        // 结果赋值
        auto &rect = bboxs[r];
        rect.x = lt.x, rect.y = lt.y;
        rect.width = wh.width, rect.height = wh.height;
        _pscore[r] = cost_sr;
        _costs[r] = score_final;
      }
    };
    cv::parallel_for_(cv::Range(0, total), fun_cal_bbox);

    // 计算最大值
    double max_score = 0;
    int idx_max = -1;
    cv::minMaxIdx(costs, nullptr, &max_score, nullptr, &idx_max);

    // 还原最终框，预测的实际上是偏差
    cv::Rect2f finalbox = bboxs[idx_max];
    float adapt_lr = _pscore[idx_max] * lr;

    cv::Rect2f predbox = recoverNewBox(cv::Size2f(cols, rows), finalbox, sz_in, scalexy, adapt_lr);

    return predbox;
  }

  void grids(const cv::Size &size, cv::Mat &gridx, cv::Mat &gridy)
  {
    int rows = size.height, cols = size.width;

    gridx.create(rows, cols, CV_32FC1), gridy.create(rows, cols, CV_32FC1);
    float *_gridx = (float *)gridx.data, *_gridy = (float *)gridy.data;

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        *_gridx = j, *_gridy = i;
        _gridx++, _gridy++;
      }
    }
  }

}