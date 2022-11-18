#include "common.h"

class YoloV5
{
public:
	YoloV5(std::string modelpath);
	int32_t initModel(std::string modelpath);
	int32_t prepareImage(std::vector<cv::Mat> &images);
	int32_t interface();
public:
	static int32_t prepareTensor(hbDNNHandle_t &dnn_handle, std::vector<hbDNNTensor> &input_tensors, std::vector<hbDNNTensor> &output_tensors);
	static int32_t cvtImageBGR2Tensor(cv::Mat &imgC, hbDNNTensor &tensor, bool padding, cv::Point2f &scalexy, cv::Vec4i &padoffset);
private:
  hbPackedDNNHandle_t packed_dnn_handle;
	hbDNNHandle_t dnn_handle;
	std::vector<hbDNNTensor> input_tensors, output_tensors;
	std::vector<cv::Point2f> scalexys;
	std::vector<cv::Vec4i> padoffsets;
};