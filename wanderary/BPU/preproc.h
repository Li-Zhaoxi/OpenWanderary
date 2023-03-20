#ifndef WDR_PREPROC_H_
#define WDR_PREPROC_H_

namespace wdr
{
namespace BPU
{

// 根据Tensor的属性，自动将输入的BGR图像，转为目标Tensor
// 目标Tensor属性仅支持BGR/RGB, HWC/CHW, NV12由于存在通道错误问题不增加相应的转换
void cvtImage2Tensor(const cv::Mat &img, hbDNNTensor &tensor);

}
}


#endif