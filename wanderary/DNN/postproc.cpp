#include <DNN/postproc.h>

namespace wdr
{
void parseBinarySegmentResult(const cv::Mat &src, std::vector<cv::Mat> &preds)
{
  // 仅支持CHW，并约束C=2
  CV_Assert(src.rows == -1 && src.cols == -1 && src.channels() == 1 && src.type() == CV_32F);
  CV_Assert(src.size.dims() == 3 || src.size.dims() == 4);
  int b, c, h, w;
  if (src.size.dims() == 3)
    b = 1, c = src.size[0], h = src.size[1], w = src.size[2];
  else
    b = src.size[0], c = src.size[1], h = src.size[2], w = src.size[3];  
  std::cout << "size: " << src.size << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c: " << c << std::endl;
  std::cout << "h: " << h << std::endl;
  std::cout << "w: " << w << std::endl; 
  // 只要通道1大于通道0即可
  CV_Assert(c == 2);
  preds.resize(b);
  for(int i = 0; i < b; i++)
  {
    preds[i].create(h, w, CV_8UC1);
    float *_bdata = ((float*)src.data) + i * c * h * w;
    float *_fdata = _bdata + h * w;
    unsigned char *_label = preds[i].data;
    
    int total_hw = h * w;
    for (int k = 0; k < total_hw; k++)
    {
      *_label = *_fdata > *_bdata ? 255 : 0;
      std::cout << *_fdata << ", " << *_bdata << ", " << int(*_label) << std::endl;
      _label++, _bdata++, _fdata++;
    }

  }
}



}