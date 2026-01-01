#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <wanderary/apps/yolo.h>
#include <wanderary/data_loader/data_loader.h>
#include <wanderary/utils/convertor.h>
#include <wanderary/utils/yaml_utils.h>
#include <wanderary/visualization/draw_boxes.h>

#include <CLI/CLI.hpp>
#include <indicators/progress_bar.hpp>

using DataLoader = wdr::loader::DataLoader;
using TimerManager = wdr::TimerManager;
using AutoScopeTimer = wdr::AutoScopeTimer;
using StatisticsTimeManager = wdr::StatisticsTimeManager;
using YOLOv8 = wdr::apps::YOLOv8;
using Box2DDrawer = wdr::vis::Box2DDrawer;
using ProgressBar = indicators::ProgressBar;

struct CropInfo {
  explicit CropInfo(const wdr::json &cfg);
  cv::Size crop_size;
  cv::Size crop_offset;
  bool drop_gap;
};

// 基于一个数据集处理YoloV8
void process_yolov8_dataset(const std::string &config_path,
                            const std::string &input_file,
                            const std::string &saveroot, int thread_num);
void process_yolov8_image(const std::string &config_path,
                          const std::string &input_file,
                          const std::string &saveroot,
                          const std::string &image_path, int thread_num);

int main(int argc, char **argv) {
  CLI::App app{"模型各阶段耗时统计工具"};
  app.require_subcommand(1);

  /* 构造YoloV8推理模型
   */
  auto yolov8 = app.add_subcommand(
      "yolov8",
      "YoloV8模型推理, "
      "需参考https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/"
      "detect/YOLOv8/YOLOv8-Detect_YUV420SP完成量化");
  yolov8->require_subcommand(1);
  std::string config_file;
  std::string input_dataset;

  int thread_num = 1;
  std::string saveroot;

  yolov8->add_option("-c,--config", config_file, "配置文件")->required();
  yolov8->add_option("-t,--thread", thread_num, "线程数")->default_val(1);
  yolov8->add_option("-s,--save-root", saveroot, "结果保存路径")
      ->default_val("");
  yolov8->add_option("-i,--input", input_dataset, "数据集输入文件")->required();

  auto yolov8_dataset = yolov8->add_subcommand("dataset", "数据集处理模式");
  yolov8_dataset->callback([&config_file, &input_dataset, &thread_num,
                            &saveroot]() {
    process_yolov8_dataset(config_file, input_dataset, saveroot, thread_num);
  });

  auto yolov8_testone = yolov8->add_subcommand("single-image", "单张测试模式");
  std::string input_image;
  yolov8_testone->add_option("--input-image", input_image, "单张图片输入文件")
      ->required();
  yolov8_testone->callback(
      [&config_file, &input_dataset, &thread_num, &saveroot, &input_image]() {
        process_yolov8_image(config_file, input_dataset, saveroot, input_image,
                             thread_num);
      });
  CLI11_PARSE(app, argc, argv);
  return 0;
}

CropInfo::CropInfo(const wdr::json &cfg) {
  const int crop_w = wdr::GetData<int>(cfg, "size_w");
  const int crop_h = wdr::GetData<int>(cfg, "size_h");
  const int offset_w = wdr::GetData<int>(cfg, "offset_w");
  const int offset_h = wdr::GetData<int>(cfg, "offset_h");
  const bool drop_gap = wdr::GetData<bool>(cfg, "drop_gap");
  this->crop_size = cv::Size(crop_w, crop_h);
  this->crop_offset = cv::Point(offset_w, offset_h);
  this->drop_gap = drop_gap;
}

std::unique_ptr<Box2DDrawer> CreateDrawer(const std::string &names_path) {
  const auto names = wdr::LoadJson(names_path);
  const int class_num = names.size();
  std::vector<std::string> class_names(class_num);
  for (const auto &item : names.items())
    class_names[std::stoi(item.key())] = item.value();
  return std::make_unique<Box2DDrawer>(class_num, class_names);
}

void process_yolov8_dataset(const std::string &config_path,
                            const std::string &input_file,
                            const std::string &saveroot, int thread_num) {
  // 提取数据输入参数
  const auto total_inputs = wdr::LoadYaml<wdr::json>(input_file);
  const auto total_config = wdr::LoadYaml<wdr::json>(config_path);

  DataLoader data_loader(wdr::GetData<wdr::json>(total_inputs, "data_loader"));
  CropInfo crop_info(wdr::GetData<wdr::json>(total_inputs, "crop_info"));
  YOLOv8 yolov8("yolov8", total_config, thread_num);

  // 初始化可视化
  std::unique_ptr<Box2DDrawer> drawer =
      (saveroot.length() > 0)
          ? CreateDrawer(wdr::GetData<std::string>(total_inputs, "type_names"))
          : nullptr;
  const int data_size = data_loader.size();
  ProgressBar bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::Fill{"="},
      indicators::option::Lead{">"},
      indicators::option::Remainder{" "},
      indicators::option::End{"]"},
      indicators::option::PostfixText{"Process dataset. Total: " +
                                      std::to_string(data_size)},
      indicators::option::ForegroundColor{indicators::Color::green},
      indicators::option::ShowPercentage{true},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  // 推理并统计耗时
  StatisticsTimeManager stats_manager;
  for (int i = 0; i < data_size; i++) {
    const auto frame = data_loader.at(i);
    TimerManager time_manager;

    // 获取输入
    cv::Mat img;
    frame.meta.image_file->data->data.copyTo(img);
    const auto &file_path = frame.meta.image_file->rawpath;

    time_manager.start("full-pipeline");
    const auto rois =
        wdr::ImageCropROIs(img.size(), crop_info.crop_size,
                           crop_info.crop_offset, crop_info.drop_gap);
    const auto box2ds = yolov8.run(img, rois, &stats_manager);
    time_manager.stop("full-pipeline");
    stats_manager.add(time_manager);

    // 可视化
    if (drawer) {
      cv::Mat vis;
      frame.meta.image_file->data->data.copyTo(vis);
      drawer->draw(box2ds, &vis);
      const std::string savepath =
          wdr::path::join(saveroot, wdr::path::basename(file_path));
      cv::imwrite(savepath, vis);
    }
    bar.set_progress((i + 1) * 100 / data_size);
  }

  stats_manager.printStatistics();
}

void process_yolov8_image(const std::string &config_path,
                          const std::string &input_file,
                          const std::string &saveroot,
                          const std::string &image_path, int thread_num) {
  // 提取数据输入参数
  const auto total_inputs = wdr::LoadYaml<wdr::json>(input_file);
  const auto total_config = wdr::LoadYaml<wdr::json>(config_path);

  CropInfo crop_info(wdr::GetData<wdr::json>(total_inputs, "crop_info"));
  YOLOv8 yolov8("yolov8", total_config, thread_num);
  const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  CHECK(!img.empty()) << "Failed to load image: " << image_path;

  // 初始化可视化
  std::unique_ptr<Box2DDrawer> drawer =
      (saveroot.length() > 0)
          ? CreateDrawer(wdr::GetData<std::string>(total_inputs, "type_names"))
          : nullptr;

  // 推理并统计耗时
  StatisticsTimeManager stats_manager;
  TimerManager time_manager;

  time_manager.start("full-pipeline");
  const auto rois =
      wdr::ImageCropROIs(img.size(), crop_info.crop_size, crop_info.crop_offset,
                         crop_info.drop_gap);
  const auto box2ds = yolov8.run(img, rois, &stats_manager);
  time_manager.stop("full-pipeline");
  stats_manager.add(time_manager);

  // 可视化
  if (drawer) {
    cv::Mat vis;
    img.copyTo(vis);
    drawer->draw(box2ds, &vis);
    const std::string savepath =
        wdr::path::join(saveroot, wdr::path::basename(image_path));
    cv::imwrite(savepath, vis);
    LOG(INFO) << "Save visualization to " << savepath;
  }

  stats_manager.printStatistics();
}
