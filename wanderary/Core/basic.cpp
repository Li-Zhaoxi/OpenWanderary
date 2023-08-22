#include <Core/core.h>
#include <unistd.h>

namespace wdr
{

  void RootRequired()
  {
    if (getuid())
      CV_Error(cv::Error::StsError, "You must use ROOT or SUDO to use these BPU functions.");
  }

  void get_rgb_image(const std::string &imgpath, cv::Mat &img)
  {
    cv::Mat imgC = cv::imread(imgpath);
    if (imgC.channels() == 1)
      cv::cvtColor(imgC, img, cv::COLOR_GRAY2RGB);
    else
      cv::cvtColor(imgC, img, cv::COLOR_BGR2RGB);
  }

  void get_bgr_image(const std::string &imgpath, cv::Mat &img)
  {
    cv::Mat imgC = cv::imread(imgpath);
    if (imgC.channels() == 1)
      cv::cvtColor(imgC, img, cv::COLOR_GRAY2BGR);
    else
      img = imgC;
  }

  std::vector<size_t> get_shape(cv::Mat &mat)
  {
    std::vector<size_t> shapes;

    if (mat.rows < 0)
    {
      for (int k = 0; k < mat.size.dims(); k++)
        shapes.push_back(mat.size[k]);
    }
    else
    {
      shapes.push_back(1);
      shapes.push_back(mat.rows);
      shapes.push_back(mat.cols);
      if (mat.channels() > 1)
        shapes.push_back(mat.channels());
    }

    return shapes;
  }

  std::vector<int> shape(const cv::Mat &mat)
  {
    std::vector<int> res;
    if (mat.rows > 0)
    {
      for (int k = 0; k < mat.size.dims(); k++)
        res.push_back(mat.size[k]);
    }
    else
    {
      res.push_back(mat.rows);
      res.push_back(mat.cols);
      if (mat.channels() > 1)
        res.push_back(mat.channels());
    }
    return res;
  }

  void imequalresize(const cv::Mat &img, const cv::Size &target_size, const cv::Scalar &pad_value, cv::Mat &pad_image)
  {
    int target_w = target_size.width, target_h = target_size.height;
    int image_h = img.rows, image_w = img.cols;
    int img_channel = img.channels();

    float scale_w = target_w * 1.0 / image_w, scale_h = target_h * 1.0 / image_h;
    float scale = std::min(scale_w, scale_h);
    int new_h = int(scale * image_h), new_w = int(scale * image_w);

    cv::Mat resize_image;
    cv::resize(img, resize_image, cv::Size(new_w, new_h));

    pad_image.create(target_h, target_w, CV_MAKETYPE(img.depth(), img_channel));
    pad_image.setTo(pad_value);

    int dw = (target_w - new_w) / 2, dh = (target_h - new_h) / 2;
    resize_image.copyTo(pad_image(cv::Rect(dw, dh, new_w, new_h)));
  }

  template <typename T>
  void _hanning(T *_d, int M)
  {
    const T c = CV_PI / (M - 1);

    for (int i = 0; i < M; i++)
    {
      _d[i] = 0.5 + 0.5 * std::cos(c * i);
    }
  }

  void hanning(int M, cv::OutputArray dst, int depth)
  {
    CV_Assert(depth == CV_32F || depth == CV_64F);
    cv::Mat mat = dst.getMat();
    if (M == 0)
      mat = cv::Mat();
    else if (M == 1)
      mat = cv::Mat::ones(1, 1, CV_MAKETYPE(depth, 1));
    else
    {
      mat.create(M, 1, CV_MAKETYPE(depth, 1));
      if (depth == CV_32F)
        _hanning<float>((float *)mat.data, M);
      else if (depth == CV_64F)
        _hanning<double>((double *)mat.data, M);
      else
        CV_Error(cv::Error::StsAssert, "Invalid mat depth: " + std::to_string(depth));
    }
  }

}