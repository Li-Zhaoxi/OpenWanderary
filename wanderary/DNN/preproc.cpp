#include <DNN/preproc.h>

namespace wdr
{
  void preprocess_onboard_NHWC(const cv::Mat img, int modelh, int modelw, cv::Mat &datain)
  {
    cv::Mat tmp;
    // Python: img = cv2.resize(img, (modelw, modelh))
    cv::resize(img, tmp, cv::Size(modelw, modelh));

    // Python: img = np.expand_dims(img, 0)
    // Python: img = np.ascontiguousarray(img)
    std::vector<int> dims = {1, tmp.rows, tmp.cols, tmp.channels()};
    datain.create(dims.size(), dims.data(), CV_MAKETYPE(img.depth(), 1));

    memcpy(datain.data, tmp.data, tmp.total() * tmp.elemSize());
  }

  void preprocess_onboard_YoloV5BGRNHWC(const cv::Mat img, int modelh, int modelw, cv::Mat &datain)
  {
    cv::Mat inputImage;
    int rows = img.rows, cols = img.cols;
    int _max = std::max(rows, cols);

    inputImage = cv::Mat::zeros(_max, _max, CV_MAKETYPE(img.depth(), img.channels()));
    img.copyTo(inputImage(cv::Rect(0, 0, cols, rows)));

    imequalresize(inputImage, cv::Size(modelw, modelh), cv::Scalar(127, 127, 127), datain);
  }
}