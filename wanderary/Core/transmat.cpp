#include <Core/core.h>

namespace wdr
{

  void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
  {
    const int src_h = src.rows();
    const int src_w = src.cols();
    const int src_c = src.channels();

    cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

    const std::array<int, 3> dims = {src_c, src_h, src_w};
    dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
    cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

    cv::transpose(hw_c, dst_1d);
  }

  void chw_to_hwc(cv::InputArray src, cv::OutputArray dst)
  {
    const auto &src_size = src.getMat().size;
    const int src_c = src_size[0];
    const int src_h = src_size[1];
    const int src_w = src_size[2];

    auto c_hw = src.getMat().reshape(0, {src_c, src_h * src_w});

    dst.create(src_h, src_w, CV_MAKETYPE(src.depth(), src_c));
    cv::Mat dst_1d = dst.getMat().reshape(src_c, {src_h, src_w});

    cv::transpose(c_hw, dst_1d);
  }

  void makeContinuous(const cv::Mat &src, cv::Mat &dst)
  {
    if (!src.isContinuous())
      src.copyTo(dst);
    else
      dst = src;
  }

  void numpy2cv(const cnpy::NpyArray &npmat, cv::OutputArray cvmat, int dtype)
  {
    std::vector<int> dims;
    for (auto dim : npmat.shape)
      dims.push_back(dim);
    cvmat.create(dims.size(), dims.data(), CV_MAKETYPE(dtype, 1));

    auto mat = cvmat.getMat();
    int dstbytes = mat.total() * mat.elemSize();
    int srcbytes = npmat.data_holder->size() * sizeof(char);
    if (dstbytes != srcbytes)
    {
      std::stringstream ss;
      ss << "bytesizes are not equal. numpy array size: " << srcbytes;
      ss << ", dst cv mat bytesize: " << dstbytes;
      CV_Error(cv::Error::StsAssert, ss.str());
    }
    memcpy(mat.data, npmat.data_holder->data(), srcbytes);
  }

}