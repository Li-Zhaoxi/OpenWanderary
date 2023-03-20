#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>

#include <BPU/bpu.h>
#include <DNN/postproc.h>

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // basic infos
  std::string binpath = "examples/modules/unet.bin";
  std::string unetfield = "unet";
  std::string imgpath = "examples/modules/CONIC_161.png";

  // cv::Mat img = cv::imread(imgpath);
  // std::cout << img.size() << std::endl;
  // std::cout << img.rows << ", " << img.cols << ", " << img.channels() << std::endl;

  // std::vector<int> dims = {1,3,256,256};
  // cv::Mat data(4, &dims[0], CV_32F);
  // std::cout << data.size() << std::endl;
  // std::cout << data.rows << ", " << data.cols << ", " << data.channels() << std::endl;

  // 1. load model and its handle
  hbPackedDNNHandle_t pPackedNets = nullptr;
  std::unordered_map<std::string, hbDNNHandle_t> netsMap;
  wdr::BPU::readNets({binpath}, pPackedNets, netsMap);

  // 2. Get Tensor Properties and Allocate BPU Memory
  hbDNNHandle_t unet = netsMap[unetfield];
  std::vector<hbDNNTensorProperties> inProperties, outProperties;
  std::vector<hbDNNTensor> inTensors, outTensors;

  wdr::BPU::readNetProperties(unet, inProperties, true);
  wdr::BPU::readNetProperties(unet, outProperties, false);
  wdr::BPU::showhbDNNTensorProperties(inProperties[0]);
  wdr::BPU::showhbDNNTensorProperties(outProperties[0]);
  wdr::BPU::createTensors(inProperties, inTensors);
  wdr::BPU::createTensors(outProperties, outTensors);

  // 3. Start Interface
  //  Read Image->Pre-processing->doInfer->Post-processing->Save Result
  cv::Mat img = cv::imread(imgpath), output;
  std::vector<cv::Mat> preds;
  
  std::vector<int> dims = {1, 256, 256, 3};
  cv::Mat imgin(4, &dims[0], CV_8UC1);
  memcpy(imgin.data, img.data, 256*256*3);

  wdr::BPU::bpuMemcpy(imgin, inTensors[0]); // only one input
  wdr::BPU::forward(unet, inTensors, outTensors);
  wdr::BPU::bpuMemcpy(outTensors[0], output);
  wdr::parseBinarySegmentResult(output, preds);
  std::cout << output.size << std::endl;

  // // 4. Save Results;
  // cv::imwrite("pred.png", preds[0]);

  // 5. Release Tensors and Models
  wdr::BPU::releaseTensors(inTensors), wdr::BPU::releaseTensors(outTensors);
  wdr::BPU::releaseNets(pPackedNets);

  google::ShutdownGoogleLogging();
  return 0;
}