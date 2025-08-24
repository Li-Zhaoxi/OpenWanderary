#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <wanderary/data_loader/data_loader.h>
#include <wanderary/dnn/bpu_nets.h>
#include <wanderary/process/process_base.h>
#include <wanderary/utils/time_manager.h>
#include <wanderary/utils/yaml_utils.h>
#include <wanderary/visualization/draw_boxes.h>

DEFINE_string(config, "", "");
DEFINE_string(inputs, "", "");

using DataLoader = wdr::loader::DataLoader;
using ProcessManager = wdr::proc::ProcessManager;
using ProcessRecorder = wdr::proc::ProcessRecorder;
using TimerManager = wdr::TimerManager;
using AutoScopeTimer = wdr::AutoScopeTimer;
using StatisticsTimeManager = wdr::StatisticsTimeManager;

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // 初始化
  const auto total_config = wdr::LoadYaml<wdr::json>(std::string(FLAGS_config));
  const auto total_inputs = wdr::LoadYaml<wdr::json>(std::string(FLAGS_inputs));
  DataLoader data_loader(wdr::GetData<wdr::json>(total_config, "data_loader"));
  data_loader.load(wdr::GetData<wdr::json>(total_inputs, "data_loader"));

  ProcessManager preproc(wdr::GetData<wdr::json>(total_config, "preprocess"));
  ProcessManager postproc(wdr::GetData<wdr::json>(total_config, "postprocess"));

  wdr::dnn::BPUNets nets(
      wdr::GetData<std::vector<std::string>>(total_inputs, "model_paths"));
  const auto model_name = wdr::GetData<std::string>(total_inputs, "model_name");
  const auto dequant_scales = nets.GetDequantScales(model_name);

  // 数据加载
  const int data_size = data_loader.size();
  std::vector<cv::Mat> out_feats;
  std::vector<wdr::Box2D> box2ds;
  StatisticsTimeManager stats_manager;
  for (int i = 0; i < data_size; i++) {
    const auto frame = data_loader.at(i);
    TimerManager time_manager;
    cv::Mat img;
    frame.meta.image_file->data->copyTo(img);
    const auto &file_path = frame.meta.image_file->rawpath;

    ProcessRecorder recorder;

    {  // 预处理
      AutoScopeTimer timer("pre-process", &time_manager);
      preproc.Forward(&img, &recorder);
    }

    {  // 推理
      AutoScopeTimer timer("bpu-inference", &time_manager);
      recorder.dequant_scales = dequant_scales;
      const std::vector<cv::Mat> input_mats = {
          cv::Mat(1, img.total(), CV_8UC1, img.data)};
      nets.Forward(model_name, input_mats, &out_feats);
    }

    {  // 后处理
      AutoScopeTimer timer("post-process", &time_manager);
      postproc.Forward(&out_feats, &box2ds, &recorder);
    }
    stats_manager.add(time_manager);

    // 可视化
    cv::Mat vis_img;
    frame.meta.image_file->data->copyTo(vis_img);
    wdr::vis::DrawBoxes2D(box2ds, &vis_img);
    cv::imwrite("build/" + std::to_string(i) + ".jpg", vis_img);
  }
  stats_manager.printStatistics();
  return 0;
}
