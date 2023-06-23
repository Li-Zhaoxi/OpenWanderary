#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <boost/filesystem.hpp>

#include <cnpy/cnpy.h> // 3rdparty

#include <BPU/bpu.h>
#include <DNN/preproc.h>
#include <DNN/postproc.h>
#include <gflags/gflags.h>

const std::string saveroot = "projects/torchdnn/data/yolov5/";
const std::string binpath = saveroot + "yolov5s.bin";
const std::string imgpath = saveroot + "20220904134315.jpg";

DEFINE_string(mode, "class", ""); // CAPI/CPPAPI/WDR

int test_class(); // 调用WDR相关Class实现推理

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  wdr::RootRequired();

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(binpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the model file:  " + binpath);

  if (!boost::filesystem::is_regular_file(boost::filesystem::path(imgpath)))
    CV_Error(cv::Error::StsAssert, "Cannot find the image file:  " + imgpath);

  std::string mode = FLAGS_mode;
  LOG(INFO) << "mode: " << mode;

  if (mode == "class")
    test_class();
  else
    CV_Error(cv::Error::StsAssert, "Unknown mode: " + mode);

  google::ShutdownGoogleLogging();
  return 0;
}

int test_class()
{
  const std::string modelname = "yolov5s";
  // 1. 加载模型
  wdr::BPU::BpuNets nets;
  nets.readNets({binpath});
  int idxmode = nets.name2index(modelname);
  LOG(INFO) << "model index: " << idxmode;
  CV_Assert(idxmode >= 0);

  // 2. 加载图像
  cv::Mat img;
  wdr::get_bgr_image(imgpath, img);
  LOG(INFO) << "Finish load bgr image";

  // 3. 内存分配
  wdr::BPU::BpuMats input_mats, output_mats;
  nets.init(idxmode, input_mats, output_mats, true);
  LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();

  // 3. 构造预处理输出，模型输入是256，256
  cv::Mat datain;
  cv::Size modsize = input_mats[0].size(false);
  LOG(INFO) << "Input model size: " << modsize;

  wdr::preprocess_onboard_YoloV5BGRNHWC(img, modsize.height, modsize.width, datain);
  input_mats[0] << datain; // datain数据拷贝到Tensor里
  input_mats.bpu();        // 更新数据到BPU中
  LOG(INFO) << "Finish preprocess";

  // 4. 模型推理
  cv::Mat dataout;
  nets.forward(idxmode, input_mats, output_mats);
  output_mats.cpu();         // 从BPU中下载数据
  output_mats[0] >> dataout; // 从Tensor里拷出数据到dataout
  LOG(INFO) << "Finish infer";

  // // 5. 构造后处理数据，并保存最终预测结果
  // std::vector<cv::Mat> preds;
  // wdr::parseBinarySegmentResult(dataout, preds);
  // for (int k = 0; k < preds.size(); k++)
  //   cv::imwrite(saveroot + "pred_cpp_wdr_" + std::to_string(k) + ".png", preds[k]);

  // 内存释放由代码的析构函数自动完成，无需主动调用

  return 0;
}