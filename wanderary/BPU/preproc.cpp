#include <BPU/bpu.h>
#include <Core/core.h>
namespace wdr
{
namespace BPU
{

void cvtImage2Tensor(const cv::Mat &img, hbDNNTensor &tensor)
{
  cv::Mat usage_img, dstlayoutmat, finalmat;

  const auto& tensor_property = tensor.properties;

  // cvt BGG to dst format
  switch(tensor_property.tensorType)
  {
    case HB_DNN_IMG_TYPE_RGB:
      cv::cvtColor(img, usage_img, cv::COLOR_BGR2RGB);
      break;
    case HB_DNN_IMG_TYPE_BGR:
      img.copyTo(usage_img);
      break;
    default:
      LOG(ERROR) << "Unsupport type: ";
      showhbDNNDataType(hbDNNDataType(tensor_property.tensorType));
      std::abort();
      break;
  }

  // cvt HWC to dst format
  switch(tensor_property.tensorLayout)
  {
    case HB_DNN_LAYOUT_NHWC:
      dstlayoutmat = usage_img;
      break;
    case HB_DNN_LAYOUT_NCHW:
      hwc_to_chw(usage_img, dstlayoutmat);
      break;
    default:
      LOG(ERROR) << "Unsupport type: ";
      showhbDNNTensorLayout(hbDNNTensorLayout(tensor_property.tensorLayout));
      std::abort();
      break;
  }

  // make memory continuous
  makeContinuous(dstlayoutmat, finalmat);

  int memsize = finalmat.total() * finalmat.elemSize();
  bpuMemcpy(tensor, (uint8_t*)finalmat.data, memsize);
}




}
}