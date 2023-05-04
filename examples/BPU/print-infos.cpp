#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

#include <BPU/bpu.h>

DEFINE_string(binpath, "", "");
DEFINE_string(mode, "class", "");

int test_class(const boost::filesystem::path &binpath);
int test_wdrapis(const boost::filesystem::path &binpath);

//// 基本信息打印功能
// 1. 打印模型个数，以及模型名称
// 2. 打印第一个模型的输入Tensor的第一个Tensor
// 3. 打印第一个模型的输出Tensor的第一个Tensor
// 4. 打印所有模型参数信息

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path binpath(FLAGS_binpath);
  std::string mode = FLAGS_mode;

  LOG(INFO) << "binpath: " << binpath << std::endl;
  LOG(INFO) << "mode: " << mode << std::endl;

  CV_Assert(boost::filesystem::is_regular_file(binpath));

  if (mode == "class")
    test_class(binpath);
  else if (mode == "wdrapi")
    test_wdrapis(binpath);
  else
    CV_Error(cv::Error::StsAssert, "Unknown mode: " + mode);

  google::ShutdownGoogleLogging();
  return 0;
}

int test_class(const boost::filesystem::path &binpath)
{
  // 0. 加载模型
  wdr::BPU::BpuNets nets;
  nets.readNets({binpath.string()}); // 加载模型

  // 1. 打印模型个数，以及模型名称
  std::stringstream ss;
  ss << "model num: " << nets.total() << ", model names: ";
  for (int k = 0; k < nets.total(); k++)
    ss << nets.index2name(k) << ", ";
  LOG(INFO) << ss.str(), ss.clear();

  // 2. 打印第一个模型的输入Tensor的第一个Tensor
  LOG(INFO) << nets[0][wdr::BPU::NET_INPUT][0];

  // 3. 打印第一个模型的输出Tensor的第一个Tensor
  LOG(INFO) << nets[0][wdr::BPU::NET_OUTPUT][0];

  // 4. 打印所有模型参数信息
  for (int k = 0; k < nets.total(); k++)
    LOG(INFO) << nets.index2name(k) << ", " << nets[k];

  return 0;
}

int test_wdrapis(const boost::filesystem::path &binpath)
{
  // 0. 加载模型
  hbPackedDNNHandle_t pPackedNets = nullptr;
  std::unordered_map<std::string, hbDNNHandle_t> netsMap;
  wdr::BPU::readNets({binpath.string()}, pPackedNets, netsMap);

  // 1. 打印模型个数，以及模型名称
  std::stringstream ss;
  ss << "model num: " << netsMap.size() << ", model names: ";
  for (const auto &net : netsMap)
    ss << net.first << ", ";
  LOG(INFO) << ss.str(), ss.clear();

  // 2. 打印第一个模型的输入Tensor的第一个Tensor
  hbDNNHandle_t handle_model = netsMap.begin()->second;
  std::vector<hbDNNTensorProperties> input_properties;
  wdr::BPU::readNetProperties(handle_model, input_properties, true);
  LOG(INFO) << input_properties[0];

  // 3. 打印第一个模型的输出Tensor的第一个Tensor
  std::vector<hbDNNTensorProperties> output_properties;
  wdr::BPU::readNetProperties(handle_model, output_properties, false);
  LOG(INFO) << output_properties[0];

  // 4. 打印所有模型参数信息
  for (const auto &net : netsMap)
  {
    hbDNNHandle_t nethandle = net.second;
    std::vector<hbDNNTensorProperties> input_infos, output_infos;
    wdr::BPU::readNetProperties(nethandle, input_infos, true);
    wdr::BPU::readNetProperties(nethandle, output_infos, false);
    ss << net.first << std::endl;
    ss << "INPUT: " << std::endl;
    for (const auto &info : input_infos)
      ss << info << std::endl;

    ss << "OUTPUT: " << std::endl;
    for (const auto &info : output_infos)
      ss << info << std::endl;
  }
  LOG(INFO) << ss.str(), ss.clear();

  // 释放模型
  wdr::BPU::releaseNets(pPackedNets);
  return 0;
}