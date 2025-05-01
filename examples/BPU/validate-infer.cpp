#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <wanderary/Core/core.h>
#include <wanderary/BPU/bpu.h>

DEFINE_string(dataroot, "", "");
DEFINE_string(filelist, "", "");

DEFINE_string(binpath, "", "");
DEFINE_string(modelname, "", "");
DEFINE_string(dtypes, "", "");

bool checkEqual(cv::InputArray mat1, cv::InputArray mat2)
{
  wdr::BPU::TensorSize size1(mat1), size2(mat2);
  if (size1 != size2)
  {
    LOG(INFO) << "size1 is " << size1 << ", is not equal to size2: " << size2;
    return false;
  }

  const cv::Mat cvmat1 = mat1.getMat(), cvmat2 = mat2.getMat();
  const int bytesize1 = cvmat1.total() * cvmat1.elemSize(), bytesize2 = cvmat2.total() * cvmat2.elemSize();
  if (bytesize1 != bytesize2)
  {
    LOG(INFO) << "Bytesizes are not equal. bytesize1: " << bytesize1 << ", bytesize2: " << bytesize2;
    return false;
  }

  const uchar *_mat1 = cvmat1.data, *_mat2 = cvmat2.data;
  for (int i = 0; i < bytesize1; i++, _mat1++, _mat2++)
  {
    if (*_mat1 != *_mat2)
      return false;
  }
  return true;
}

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  wdr::RootRequired();

  //// 1. 加载文件列表
  std::vector<std::string> inferpaths;
  {
    std::string dataroot = FLAGS_dataroot;
    std::string filelistpath = FLAGS_filelist;
    LOG(INFO) << "dataroot: " << dataroot;
    LOG(INFO) << "filelistpath: " << filelistpath;
    CV_Assert(wdr::path::exist(dataroot, true) && wdr::path::exist(filelistpath, true));

    // 构造路径列表
    std::ifstream infile(filelistpath);
    while (!infile.eof())
    {
      std::string str;
      infile >> str;
      if (str.length() == 0)
        break;
      std::string filepath = wdr::path::join({dataroot, str});
      if (wdr::path::exist(filepath))
        inferpaths.push_back(filepath);
      else
        LOG(WARNING) << "Cannot find the path: " << filepath;
    }
  }

  //// 2. 初始化BPU模型
  wdr::BPU::BpuNets nets;
  wdr::BPU::BpuMats input_mats, output_mats;
  int idxmode = -1;
  {
    // Load model
    std::string binpath = FLAGS_binpath;
    std::string modelname = FLAGS_modelname;
    LOG(INFO) << "binpath: " << binpath;
    LOG(INFO) << "modelname: " << modelname;
    CV_Assert(wdr::path::exist(binpath, true));
    nets.readNets({binpath}); // 加载模型

    // init model
    idxmode = nets.name2index(modelname);
    LOG(INFO) << "model index: " << idxmode;
    CV_Assert(idxmode >= 0);
    nets.init(idxmode, input_mats, output_mats, true);
  }

  //// 3. 识别数据类型
  const int inputnum = input_mats.size(), outputnum = output_mats.size();
  std::vector<int> input_types, output_types;
  {
    std::string strtypes = FLAGS_dtypes;
    LOG(INFO) << "strtypes: " << strtypes;
    std::vector<std::string> splitdtypes = wdr::argparse::split(strtypes, ",");
    if (splitdtypes.size() != (inputnum + outputnum))
    {
      std::stringstream ss;
      ss << "The input/output number is " << splitdtypes.size();
      ss << ", is not equal to " << inputnum + outputnum;
      CV_Error(cv::Error::StsAssert, ss.str());
    }
    LOG(INFO) << "Recognize " << splitdtypes.size() << " types";
    for (int idx = 0; idx < inputnum; idx++)
      input_types.push_back(wdr::stot(splitdtypes[idx]));
    for (int idx = inputnum; idx < (inputnum + outputnum); idx++)
      output_types.push_back(wdr::stot(splitdtypes[idx]));
  }

  //// 4. 遍历所有npz文件，开始测试推理
  for (int idxfile = 0; idxfile < inferpaths.size(); idxfile++)
  {
    const auto &filepath = inferpaths[idxfile];
    if (!wdr::path::exist(filepath))
    {
      LOG(WARNING) << "Cannot find the infer path: " << filepath;
      continue;
    }
    else
      LOG(INFO) << "Processing " << idxfile << "th file: " << filepath;

    // (1) numpy矩阵转opencv的输入输出矩阵
    cnpy::npz_t ioarrays = cnpy::npz_load(filepath);
    bool is_converted = true;
    std::vector<cv::Mat> cvmatsin(inputnum), cvmatsout(outputnum);
    for (int idxin = 0; idxin < inputnum && is_converted; idxin++)
    {
      std::string field = "input_" + std::to_string(idxin);
      if (ioarrays.find(field) == ioarrays.end())
      {
        is_converted = false;
        LOG(WARNING) << "Cannot find [" << field << "] in " << filepath;
      }
      else
      {
        LOG(INFO) << "Start Convert numpy array [" << field << "], type: " << input_types[idxin];
        wdr::numpy2cv(ioarrays[field], cvmatsin[idxin], input_types[idxin]);
      }
    }
    for (int idxout = 0; idxout < outputnum && is_converted; idxout++)
    {
      std::string field = "output_" + std::to_string(idxout);
      if (ioarrays.find(field) == ioarrays.end())
      {
        is_converted = false;
        LOG(WARNING) << "Cannot find [" << field << "] in " << filepath;
      }
      else
      {
        LOG(INFO) << "Start Convert numpy array [" << field << "], type: " << input_types[idxout];
        wdr::numpy2cv(ioarrays[field], cvmatsout[idxout], output_types[idxout]);
      }
    }
    if (!is_converted)
      continue;

    // (2) 数据拷入
    for (int idxin = 0; idxin < inputnum; idxin++)
    {
      auto bpumat = input_mats[idxin];
      bpumat << cvmatsin[idxin];
      bpumat.bpu();
    }

    // (3) 开始推理
    double t1 = cv::getTickCount();
    nets.forward(idxmode, input_mats, output_mats);
    double t2 = cv::getTickCount();
    LOG(INFO) << "time consumption of BPU infer: " << (t2 - t1) * 1000 / cv::getTickFrequency() << " ms";
    // (4) 验证输出
    output_mats.cpu();
    for (int idxout = 0; idxout < outputnum; idxout++)
    {
      cv::Mat tmp;
      output_mats[idxout] >> tmp;
      bool checkres = checkEqual(tmp, cvmatsout[idxout]);
      if (!checkres)
        LOG(INFO) << "NOT MATCH. idxout: " << idxout;
    }
    LOG(INFO) << "Output Checking [OK]";
  }

  // 补充新功能
  // 1. 进度条
  // 2. SLAM去畸变
  // 3. MVE部分代码重构
  // 4. VIO SLAM
  // 5. C++ Tello
  // 6. C++ Waveshare Dog
  // 7. 图像匹配网络
  // 8. 自定义层
  // 9. YoloV5 6.2版本的部署
  // 10. YoloV5 简化版本的部署
  // 11. YoloV8的部署

  /* 析构部分*/
  /* 内存释放由代码的析构函数自动完成，无需主动调用 */
  google::ShutdownGoogleLogging();
  return 0;
}