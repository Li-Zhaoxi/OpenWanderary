#include "wanderary/utils/convertor.h"

#include <glog/logging.h>

namespace wdr {

void BGRToNV12(const cv::Mat &bgr, cv::Mat *nv12) {
  CHECK_EQ(bgr.channels(), 3);
  cv::Mat yuv420p;

  // wxhx3转为wxhx3/2
  cv::cvtColor(bgr, yuv420p, cv::COLOR_BGR2YUV_I420);
  const int nv12_total = yuv420p.cols * yuv420p.rows;
  yuv420p = yuv420p.reshape(1, 1);  // 1xnv12_total

  // 分配空间
  nv12->create(1, nv12_total, CV_8UC1);

  // 拷贝y通道
  const int ysize = bgr.cols * bgr.rows;
  yuv420p.colRange(0, ysize).copyTo(nv12->colRange(0, ysize));

  // 拷贝uv通道
  cv::Mat uv_planar = yuv420p.colRange(ysize, nv12_total).reshape(1, 2).t();
  cv::Mat uv_packed = uv_planar.reshape(1, 1);
  uv_packed.copyTo(nv12->colRange(ysize, nv12_total));
}

void NV12ToYUV444(const cv::Mat &nv12, const cv::Size size, cv::Mat *yuv444) {
  const int nv12_total = nv12.cols;
  CHECK_EQ(nv12.rows, 1);
  CHECK_EQ(nv12_total, size.width * size.height * 3 / 2);
  const int ysize = size.width * size.height;

  yuv444->create(size, CV_8UC3);

  // 拷贝y通道
  cv::Mat y, u, v;
  y = nv12.colRange(0, ysize).reshape(1, size.height);

  u.create(size.height / 2, size.width / 2, CV_8UC1);
  v.create(size.height / 2, size.width / 2, CV_8UC1);

  const uchar *data = nv12.data + ysize;
  uchar *udata = u.data;
  uchar *vdata = v.data;
  for (int i = 0; i < nv12_total - ysize; i++) {
    if (i % 2 == 0)
      udata[i / 2] = data[i];
    else
      vdata[i / 2] = data[i];
  }

  cv::resize(u, u, size);
  cv::resize(v, v, size);

  cv::merge(std::vector<cv::Mat>{y, u, v}, *yuv444);
}

}  // namespace wdr
