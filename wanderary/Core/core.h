#ifndef WDR_CORE_H_
#define WDR_CORE_H_

#include <cnpy/cnpy.h>
#include <opencv2/opencv.hpp>

namespace wdr
{
  // 维度排布检查，前3位记录dims，后5位标记排布
  enum CV_MAT_LAYOUT
  {
    // Not Check
    LAYOUT_NONE = 0b00000000,
    // 1 dim
    LAYOUT_A = 0b00100001,
    // 2 dims
    LAYOUT_AB = 0b01000000,
    LAYOUT_A1 = LAYOUT_AB + 1,
    LAYOUT_1B = LAYOUT_AB + 2,
    // 3 dims
    LAYOUT_ABC = 0b01100000,
    LAYOUT_AB1 = LAYOUT_ABC + 1,
    LAYOUT_A1C = LAYOUT_ABC + 2,
    LAYOUT_1BC = LAYOUT_ABC + 3,
    LAYOUT_A11 = LAYOUT_ABC + 4,
    LAYOUT_1B1 = LAYOUT_ABC + 5,
    LAYOUT_11C = LAYOUT_ABC + 6,
    // 4 dims
    LAYOUT_ABCD = 0b10000000,
    LAYOUT_ABC1 = LAYOUT_ABCD + 1,
    LAYOUT_AB1D = LAYOUT_ABCD + 2,
    LAYOUT_A1CD = LAYOUT_ABCD + 3,
    LAYOUT_1BCD = LAYOUT_ABCD + 4,
    LAYOUT_AB11 = LAYOUT_ABCD + 5,
    LAYOUT_A11D = LAYOUT_ABCD + 6,
    LAYOUT_11CD = LAYOUT_ABCD + 7,
    LAYOUT_A1C1 = LAYOUT_ABCD + 8,
    LAYOUT_1B1D = LAYOUT_ABCD + 9,
    LAYOUT_1BC1 = LAYOUT_ABCD + 10,
    LAYOUT_A111 = LAYOUT_ABCD + 11,
    LAYOUT_1B11 = LAYOUT_ABCD + 12,
    LAYOUT_11C1 = LAYOUT_ABCD + 13,
    LAYOUT_111D = LAYOUT_ABCD + 14
  };

  template <typename T, typename iterT>
  inline bool in(const T &val, const iterT &candidates)
  {
    for (auto &v : candidates)
      if (val == v)
        return true;
    return false;
  }

  // logs.cpp: 记录各种可视化，以及检查项
  std::string dtype(cv::InputArray src);
  int stot(const std::string &strtype);
  std::string layout(CV_MAT_LAYOUT lay);
  bool MatCheck(const cv::Mat &mat, std::string &errmsg, const std::vector<CV_MAT_LAYOUT> &layouts = {}, int channels = -1, const std::vector<int> &dtypes = {}, bool continuous = false);

  ////////////// basic
  void RootRequired();

  void get_rgb_image(const std::string &imgpath, cv::Mat &img);
  void get_bgr_image(const std::string &imgpath, cv::Mat &img);
  std::vector<size_t> get_shape(cv::Mat &mat);
  std::vector<int> shape(const cv::Mat &mat);

  template <typename T>
  std::vector<T> squeeze(const std::vector<T> &dims)
  {
    std::vector<T> res;
    for (auto dim : dims)
    {
      if (dim != 1)
        res.push_back(dim);
    }
    return res;
  }

  void imequalresize(const cv::Mat &img, const cv::Size &target_size, const cv::Scalar &pad_value, cv::Mat &pad_image);

  // class

  // refer to https://answers.opencv.org/question/226929/how-could-i-change-memory-layout-from-hwc-to-chw/
  void hwc_to_chw(cv::InputArray src, cv::OutputArray dst);
  void chw_to_hwc(cv::InputArray src, cv::OutputArray dst);
  void makeContinuous(const cv::Mat &src, cv::Mat &dst);

  template <typename T>
  inline T fast_exp(T x)
  {
    union
    {
      uint32_t i;
      T f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
  }

  template <typename T>
  void sigmode(T *_data, int num)
  {
    for (int i = 0; i < num; i++, _data++)
      *_data = T(1.0) / (T(1.0) + fast_exp(-*_data));
  }

  template <typename T>
  inline T sigmode(T data)
  {
    // return T(1.0) / (T(1.0) + fast_exp(-data));
    return T(1.0) / (T(1.0) + std::exp(-data));
  }

  void numpy2cv(const cnpy::NpyArray &npmat, cv::OutputArray cvmat, int dtype);

  void hanning(int M, cv::OutputArray dst, int depth);

  namespace argparse
  {
    std::vector<std::string> split(const std::string &strlist, const std::string &strsign);
  }

  namespace path
  {
    bool exist(const std::string &filepath, bool log = false);
    std::string join(const std::vector<std::string> &filepaths);
    std::string dirname(const std::string &filepath);

  }

  namespace logs
  {
    class Progress
    {
    public:
    private:
    };
  }

}

#endif