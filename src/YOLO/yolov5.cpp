#include "YOLO/yolov5.h"

YoloV5::YoloV5(std::string modelpath)
{

}

int32_t YoloV5::initModel(std::string modelpath)
{
	const char* model_name = modelpath.c_str();
	HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_name, 1),
      "hbDNNInitializeFromFiles failed");

	const char **model_name_list;
	int model_count = 0;
	HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
      "hbDNNGetModelNameList failed");
	
	HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
      "hbDNNGetModelHandle failed");

	return prepareTensor(this->dnn_handle, this->input_tensors, this->output_tensors);
}

int32_t YoloV5::prepareTensor(hbDNNHandle_t &dnn_handle, std::vector<hbDNNTensor> &input_tensors, std::vector<hbDNNTensor> &output_tensors)
{
	int input_count = 0, output_count = 0;
	HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "hbDNNGetInputCount failed");
	HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "hbDNNGetOutputCount failed");

	std::cout << "input_count: " << input_count << ", output_count: " << output_count << std::endl;

	input_tensors.resize(input_count), output_tensors.resize(output_count);
	
	for(int i = 0; i < input_count; i++)
	{
		HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");

		std::cout << "input_tensors: " << input_tensors[i].properties << std::endl;
		CV_Assert(0);
		int input_memSize = input_tensors[i].properties.alignedByteSize;
		// std::cout << "input_memSize: " << input_memSize << std::endl;
		// std::cout << "tensorType: " << input_tensors[i].properties.tensorType << std::endl;
		// std::cout << "tensorLayout: " << input_tensors[i].properties.tensorLayout << std::endl;
		HB_CHECK_SUCCESS(hbSysAllocCachedMem(&input_tensors[i].sysMem[0], input_memSize), "hbSysAllocCachedMem failed");

		/** Tips:
     * For input tensor, aligned shape should always be equal to the real
     * shape of the user's data. If you are going to set your input data with
     * padding, this step is not necessary.
     * */
		input_tensors[i].properties.alignedShape = input_tensors[i].properties.validShape;
	}

	for(int i = 0; i < output_count; i++)
	{
		HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
		std::cout << "output_tensors: " << output_tensors[i].properties << std::endl;
		assert(0);
    	int output_memSize = output_tensors[i].properties.alignedByteSize;
		// std::cout << "output_memSize: " << output_memSize << std::endl;
		// std::cout << "tensorType: " << output_tensors[i].properties.tensorType << std::endl;
		// std::cout << "tensorLayout: " << output_tensors[i].properties.tensorLayout << std::endl;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output_tensors[i].sysMem[0], output_memSize), "hbSysAllocCachedMem failed");
	}
	return 0;
}

int32_t YoloV5::prepareImage(std::vector<cv::Mat> &images)
{
	const int input_num = input_tensors.size();
	std::cout << "input_num: " << input_num << ", images size: " << images.size() <<std::endl;
	CV_Assert(input_num == images.size());

	scalexys.resize(input_num), padoffsets.resize(input_num);
	for(int i = 0; i < input_num; i++)
	{
		auto &usage_tensor = input_tensors[i];
		auto &usage_image = images[i];
		if (usage_tensor.properties.tensorType == HB_DNN_IMG_TYPE_BGR)
		{
			std::cout << "usage_tensor.properties.tensorType == HB_DNN_IMG_TYPE_BGR" << std::endl;
			HB_CHECK_SUCCESS(cvtImageBGR2Tensor(usage_image, usage_tensor, true, scalexys[i], padoffsets[i]),
        "cvtImageBGR2Tensor failed");
		}
		else
		{
			std::cout << "usage_tensor.properties.tensorType: " << usage_tensor.properties.tensorType << std::endl;
			return -1;
		}
	}

	return 0;

}

int32_t YoloV5::cvtImageBGR2Tensor(cv::Mat &imgC, hbDNNTensor &tensor, bool padding, cv::Point2f &scalexy, cv::Vec4i &padoffset)
{
	if (imgC.empty()) {
    LOG(ERROR) << "imgC is empty";
    return -1;
  }
	cv::Mat imgT, imgRes;
	if (imgC.channels() == 1)
		cv::cvtColor(imgC, imgT, cv::COLOR_GRAY2BGR);
	else
		imgT = imgC;
	
	auto &Properties = tensor.properties;
	int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
	if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }
	if (input_h % 2 || input_w % 2) {
    LOG(ERROR) << "input img height and width must aligned by 2!";
    return -1;
  }
	
	double scale_x = double(input_w) / imgT.cols;
	double scale_y = double(input_h) / imgT.rows;
	if (padding)
	{
		const int ori_rows = imgT.rows, ori_cols = imgT.cols;
		
		double scale = std::min(scale_x, scale_y);

		int new_rows = int(scale * ori_rows + 0.5), new_cols = int(scale * ori_cols + 0.5);
		cv::resize(imgT, imgRes, cv::Size(new_cols, new_rows));

		scalexy.x = scale, scalexy.y = scale;
		int diff_rows = input_h - new_rows, diff_cols = input_w - new_cols;
		int pad_top = diff_rows / 2, pad_bottom = diff_rows - pad_top;
		int pad_left = diff_cols / 2, pad_right = diff_cols - pad_left;

		cv::copyMakeBorder(imgRes, imgRes, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT);

	}
	else
	{
		cv::resize(imgT, imgRes, cv::Size(input_w, input_h));
		padoffset = cv::Vec4i(0, 0, 0, 0);
		scalexy.x = scale_x, scalexy.y = scale_y;
	}

	// copy bgr data to input_tensor
	auto data = tensor.sysMem[0].virAddr;
	int32_t _size = input_h * input_w * 3;
	memcpy(reinterpret_cast<uint8_t *>(data), imgRes.data, _size);

	HB_CHECK_SUCCESS(hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem failed");
	return 0;
}


int32_t YoloV5::interface()
{
	const int input_count = input_tensors.size(), output_count = output_tensors.size();

	hbDNNInferCtrlParam infer_ctrl_param;
	hbDNNTaskHandle_t task_handle = nullptr;
	HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
	
	auto _poutput_tensors = output_tensors.data();
	HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &_poutput_tensors, input_tensors.data(),
                              dnn_handle, &infer_ctrl_param), "hbDNNInfer failed");

	// wait task done 等待任务结束
	HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone failed");
	
	// release task handle
	HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");

	return 0;
}