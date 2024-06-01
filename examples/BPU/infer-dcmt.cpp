#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include <cnpy/cnpy.h>
#include <wanderary/Algorithms/tracker.h>

bool load_infer_data(cnpy::npz_t &ioarrays,
                     int input_num, const std::vector<int> &input_types,
                     int output_num, const std::vector<int> &output_types,
                     std::vector<cv::Mat> &input_data, std::vector<cv::Mat> &output_data)
{
  bool is_converted = true;
  input_data.resize(input_num), output_data.resize(output_num);

  for (int idxin = 0; idxin < input_num && is_converted; idxin++)
  {
    std::string field = "input_" + std::to_string(idxin);
    if (ioarrays.find(field) == ioarrays.end())
    {
      is_converted = false;
      LOG(WARNING) << "Cannot find [" << field << "]";
    }
    else
    {
      LOG(INFO) << "Start Convert numpy array [" << field << "], type: " << input_types[idxin];
      wdr::numpy2cv(ioarrays[field], input_data[idxin], input_types[idxin]);
    }
  }

  for (int idxout = 0; idxout < output_num && is_converted; idxout++)
  {
    std::string field = "output_" + std::to_string(idxout);
    if (ioarrays.find(field) == ioarrays.end())
    {
      is_converted = false;
      LOG(WARNING) << "Cannot find [" << field << "]";
    }
    else
    {
      LOG(INFO) << "Start Convert numpy array [" << field << "], type: " << input_types[idxout];
      wdr::numpy2cv(ioarrays[field], output_data[idxout], output_types[idxout]);
    }
  }

  return is_converted;
}

int test_infer();
int test_wholeinfer();
int track_sequence();

int track_sequence_capi();

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // test_infer();
  // test_wholeinfer();
  // track_sequence();
  track_sequence_capi();

  google::ShutdownGoogleLogging();
  return 0;
}

int test_wholeinfer()
{
  std::string binpath = "projects/torchdnn/data/dcmt/DCMT.bin";
  cv::Rect2f target(653, 221, 55, 40);
  std::string imgpath = "projects/torchdnn/data/dcmt/ChasingDrones/00001.jpg";
  std::string trackimgpath = "projects/torchdnn/data/dcmt/ChasingDrones/00008.jpg";
  std::string npzpath = "projects/torchdnn/data/dcmt/pred.npz";

  cv::Mat img = cv::imread(imgpath);
  cv::Mat imgtrack = cv::imread(trackimgpath);

  cv::Mat cls_score, bbox_pred;
  {
    cnpy::NpyArray npzcls_score = cnpy::npz_load(npzpath, "cls_score");
    cnpy::NpyArray npzbbox_pred = cnpy::npz_load(npzpath, "bbox_pred");
    {
      std::vector<int> clsshape;
      for (auto dim : npzcls_score.shape)
      {
        clsshape.push_back(dim);
      }

      cls_score.create(clsshape.size(), clsshape.data(), CV_32FC1);
      memcpy(cls_score.data, npzcls_score.data_holder->data(), npzcls_score.data_holder->size());
      LOG(INFO) << "npzcls_score.data_holder->size(): " << npzcls_score.data_holder->size() << ", total bytes: " << cls_score.total() * cls_score.elemSize1();

      const int total = cls_score.total();
      float *_cls_score = (float *)cls_score.data;
      // std::cout << "cls_score: " << std::endl;
      // for (int k = 0; k < total; k++)
      //   std::cout << _cls_score[k] << std::endl;
    }
    {
      std::vector<int> bboxshape;
      for (auto dim : npzbbox_pred.shape)
      {
        bboxshape.push_back(dim);
      }

      bbox_pred.create(bboxshape.size(), bboxshape.data(), CV_32FC1);
      memcpy(bbox_pred.data, npzbbox_pred.data_holder->data(), npzbbox_pred.data_holder->size());
    }
  }

  wdr::DCMTConfigs dcmtcfg;
  wdr::TrackerDCMT tracker(dcmtcfg, binpath, "DCMT");

  tracker.init(img, target);
  for (int i = 0; i < 20; i++)
  {
    LOG(INFO) << "i: " << i;
    tracker.track(imgtrack);
  }

  // tracker.track(imgtrack);

  cv::Rect trackrect = tracker.get_rect();
  // tracker.track(imgtrack, cls_score, bbox_pred);

  cv::rectangle(img, cv::Rect(target.x, target.y, target.width, target.height), cv::Scalar(0, 0, 255), 2);
  cv::rectangle(imgtrack, trackrect, cv::Scalar(0, 0, 255), 2);
  cv::imwrite("img.jpg", img);
  cv::imwrite("imgtrack.jpg", imgtrack);
  return 0;
}

int test_infer()
{
  std::string binpath = "projects/torchdnn/data/dcmt/DCMT.bin";
  cv::Rect2f target(653, 221, 55, 40);
  std::string imgpath = "projects/torchdnn/data/dcmt/ChasingDrones/00001.jpg";
  std::string trackimgpath = "projects/torchdnn/data/dcmt/ChasingDrones/00002.jpg";
  std::string npzpath = "projects/torchdnn/data/dcmt/debug_infer.npz";

  cnpy::NpyArray npzx = cnpy::npz_load(npzpath, "x");
  cnpy::NpyArray npzz = cnpy::npz_load(npzpath, "z");
  cnpy::NpyArray npzz_bbox = cnpy::npz_load(npzpath, "z_bbox");
  cnpy::NpyArray npzscore = cnpy::npz_load(npzpath, "score");
  cnpy::NpyArray npzbbox = cnpy::npz_load(npzpath, "bbox");
  cv::Mat x, z, z_bbox, score, bbox;
  wdr::numpy2cv(npzx, x, CV_8U);
  wdr::numpy2cv(npzz, z, CV_8U);
  wdr::numpy2cv(npzz_bbox, z_bbox, CV_32F);
  wdr::numpy2cv(npzscore, score, CV_32F);
  wdr::numpy2cv(npzbbox, bbox, CV_32F);

  LOG(INFO) << "x: " << wdr::BPU::TensorSize(wdr::shape(x)) << ", bytesize: " << x.total() * x.elemSize();
  LOG(INFO) << "z: " << wdr::BPU::TensorSize(wdr::shape(z)) << ", bytesize: " << z.total() * z.elemSize();
  LOG(INFO) << "z_bbox: " << wdr::BPU::TensorSize(wdr::shape(z_bbox)) << ", bytesize: " << z_bbox.total() * z_bbox.elemSize();
  // LOG(INFO) << "x: " << wdr::BPU::TensorSize(wdr::shape(x)) << ", bytesize: " << x.total() * x.elemSize();
  // LOG(INFO) << "x: " << wdr::BPU::TensorSize(wdr::shape(x)) << ", bytesize: " << x.total() * x.elemSize();

  const std::string modelname = "DCMT";
  // 1. 加载模型
  wdr::BPU::BpuNets nets;
  nets.readNets({binpath});
  int idxmode = nets.name2index(modelname);
  LOG(INFO) << "model index: " << idxmode;
  CV_Assert(idxmode >= 0);

  // 3. 内存分配
  wdr::BPU::BpuMats input_mats, output_mats;
  nets.init(idxmode, input_mats, output_mats, true);
  LOG(INFO) << "input tensor num: " << input_mats.size() << ", output tensor num: " << output_mats.size();

  // //
  // input_mats[0] << x;
  // input_mats[1] << z;
  // input_mats[2] << z_bbox;
  // input_mats.bpu();
  // LOG(INFO) << "Finish preprocess";

  wdr::BPU::BpuMat bpu_z_bgr = input_mats[1];
  bpu_z_bgr << z;
  bpu_z_bgr.bpu();

  wdr::BPU::BpuMat bpu_z_box = input_mats[2];
  bpu_z_box << z_bbox, bpu_z_box.bpu();

  wdr::BPU::BpuMat bpu_crop = input_mats[0];
  bpu_crop << x, bpu_crop.bpu();
  nets.forward(idxmode, input_mats, output_mats);

  return 0;

  // -----------------模型加载部分--------------------
  // 1. 加载BIN模型集
  hbPackedDNNHandle_t packed_dnn_handle; // 模型集合指针
  const char *model_file_name = binpath.c_str();
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);

  // 2. 提取模型集中所有的模型名称
  const char **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
  for (int k = 0; k < model_count; k++) // 输出提取出的所有模型的名称
    LOG(INFO) << "Parsed Model Name: " << std::string(model_name_list[k]);
  // 3. 利用目标模型名提取模型指针
  hbDNNHandle_t dnn_handle; // ※模型指针
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // -----------------输入输出内存分配--------------------
  // 1. 获取输入/输出Tensor个数
  int input_tensornum = 0, output_tensornum = 0;
  hbDNNGetInputCount(&input_tensornum, dnn_handle);
  hbDNNGetOutputCount(&output_tensornum, dnn_handle);
  LOG(INFO) << "input tensor num: " << input_tensornum << ", output tensor num: " << output_tensornum;

  // 2. 获取输入/输出Tensor参数
  std::vector<hbDNNTensorProperties> input_properties, output_properties; // 输入/输出Tensor参数
  input_properties.resize(input_tensornum), output_properties.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
    hbDNNGetInputTensorProperties(&input_properties[k], dnn_handle, k);
  for (int k = 0; k < output_tensornum; k++)
    hbDNNGetOutputTensorProperties(&output_properties[k], dnn_handle, k);

  // 3. 利用参数分配Tensor内存
  std::vector<hbDNNTensor> input_tensors, output_tensors; // ※输入/输出Tensor
  input_tensors.resize(input_tensornum), output_tensors.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
  {
    const auto &property = input_properties[k];
    input_tensors[k].properties = property;
    input_tensors[k].properties.alignedShape = input_tensors[k].properties.validShape;
    hbSysAllocCachedMem(&input_tensors[k].sysMem[0], property.alignedByteSize);
  }
  for (int k = 0; k < output_tensornum; k++)
  {
    const auto &property = output_properties[k];
    output_tensors[k].properties = property;
    hbSysAllocCachedMem(&output_tensors[k].sysMem[0], property.alignedByteSize);
  }
  LOG(INFO) << "Finish initializing input/output tensors";

  // 测试Input输入数据拷贝
  {
    auto &tensor = input_tensors[0];
    wdr::BPU::bpuMemcpy(x, tensor, false);
    hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }
  {
    auto &tensor = input_tensors[1];
    wdr::BPU::bpuMemcpy(z, tensor, false);
    hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }
  {
    auto &tensor = input_tensors[2];
    wdr::BPU::bpuMemcpy(z_bbox, tensor, false);
    hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }

  // 4. 推理模型

  hbDNNTaskHandle_t task_handle = nullptr; // 任务句柄
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  auto ptr_outtensor = output_tensors.data();
  LOG(INFO) << "Start hbDNNInfer";
  hbDNNInfer(&task_handle, &ptr_outtensor, input_tensors.data(), dnn_handle, &infer_ctrl_param);
  LOG(INFO) << "Start hbDNNWaitTaskDone";
  // 5. 等待任务结束
  hbDNNWaitTaskDone(task_handle, 0);
  // 6. 释放任务
  LOG(INFO) << "Start hbDNNReleaseTask";
  hbDNNReleaseTask(task_handle);

  //////////// 模型推理：预处理→BPU推理→后处理 ////////////
  // 1. 释放内存
  for (auto &input : input_tensors)
    hbSysFreeMem(&(input.sysMem[0]));
  for (auto &output : output_tensors)
    hbSysFreeMem(&(output.sysMem[0]));

  // 2. 释放模型
  hbDNNRelease(packed_dnn_handle);

  return 0;
}

int track_sequence()
{
  std::string binpath = "projects/torchdnn/data/dcmt/DCMT.bin";
  cv::Rect2f target(653, 220, 55, 40);
  std::string imglistpath = "projects/torchdnn/data/dcmt/ChasingDrones/filelist.txt";
  std::string saveroot = "projects/torchdnn/data/dcmt/ChasingDrones_Results";
  std::string savemode = "video";

  // 如果有这个路径，则进行debug模式
  // std::string debugroot = "projects/torchdnn/data/dcmt/debug_infer";
  std::string debugroot = "";

  // 1. 提取图像列表
  std::string seqroot = wdr::path::dirname(imglistpath);
  std::vector<std::string> imgnames;
  std::ifstream infile(imglistpath);
  while (!infile.eof())
  {
    std::string tmp;
    infile >> tmp;
    imgnames.push_back(tmp);
    // LOG(INFO) << tmp;
  }

  // 2. 算法初始化
  std::shared_ptr<cv::VideoWriter> writer;

  wdr::DCMTConfigs dcmtcfg;
  wdr::TrackerDCMT tracker(dcmtcfg, binpath, "DCMT");

  // 3. 开始跟踪
  const int imgnum = imgnames.size();
  bool needinit = true;
  bool debugmode = debugroot.length() > 0;
  for (int idximg = 0; idximg < imgnum; idximg++)
  {
    auto &imgname = imgnames[idximg];
    LOG(INFO) << "Processing: " << imgname;

    // 加载图像
    std::string imgpath = wdr::path::join({seqroot, imgname});
    if (!wdr::path::exist(imgpath, true))
      continue;
    cv::Mat img = cv::imread(imgpath);
    if (img.empty())
    {
      LOG(INFO) << "The image is empty, check the valid of : " << imgpath;
      continue;
    }

    // 初始化视频保存器
    if (savemode == "video" && writer.get() == nullptr)
    {
      std::string videopath = wdr::path::join({saveroot, "video.mp4"});
      LOG(INFO) << "videopath: " << videopath;
      writer = std::make_shared<cv::VideoWriter>(cv::VideoWriter(videopath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10, img.size()));
      CV_Assert(writer->isOpened());
    }

    // 加载debug数据
    cnpy::npz_t ioarrays;
    if (debugmode)
    {
      std::string debugpath = wdr::path::join({debugroot, imgname + ".npz"});
      if (!wdr::path::exist(debugpath, false))
      {
        LOG(INFO) << "The debug path is not exist: " << debugpath;
        continue;
      }
      ioarrays = cnpy::npz_load(debugpath);
    }

    double t1 = cv::getTickCount();

    // 校验模式
    if (debugmode)
    {
      cv::Mat target_pos_in, target_sz_in, crop_info;
      std::vector<cv::Mat> cvmatsin, cvmatsout;
      cv::Mat target_pos_out, target_sz_out;

      // 加载基本推理数据
      bool is_getiomats = load_infer_data(ioarrays,
                                          3, {CV_8U, CV_8U, CV_32F},
                                          2, {CV_32F, CV_32F}, cvmatsin, cvmatsout);
      if (!is_getiomats)
      {
        LOG(WARNING) << "The infer data is not correct, please check the infer data";
        continue;
      }

      // 1. 校验初始过程切分是否正确
      if (ioarrays.find("target_pos_in") == ioarrays.end() ||
          ioarrays.find("target_sz_in") == ioarrays.end() ||
          ioarrays.find("crop_info") == ioarrays.end())
      {
        wdr::numpy2cv(ioarrays["target_pos_in"], target_pos_in, CV_32F);
        wdr::numpy2cv(ioarrays["target_sz_in"], target_sz_in, CV_32F);
        wdr::numpy2cv(ioarrays["crop_info"], crop_info, CV_32F);
        // LOG(INFO) << "target_pos_in: " << target_pos_in;
        // LOG(INFO) << "target_sz_in: " << target_sz_in;
        // LOG(INFO) << "crop_info: " << crop_info;
        // exit(0);
      }

      // 2. 校验后处理计算是否正确
      if (ioarrays.find("target_pos_out") == ioarrays.end() ||
          ioarrays.find("target_sz_out") == ioarrays.end())
      {
        wdr::numpy2cv(ioarrays["target_pos_out"], target_pos_out, CV_32F);
        wdr::numpy2cv(ioarrays["target_sz_out"], target_sz_out, CV_32F);
      }
    }
    else
    {
      if (needinit)
      {
        tracker.init(img, target);
        needinit = false;
      }
      else
      {
        tracker.track(img);
      }
    }

    double t2 = cv::getTickCount();
    double timeusage = (t2 - t1) * 1000 / cv::getTickFrequency();

    // draw result
    cv::Rect trackrect = tracker.get_rect();

    cv::rectangle(img, trackrect, cv::Scalar(0, 0, 255), 2);
    cv::putText(img, "Method: DCMT [RDK X3, C++]", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
    cv::putText(img, "Whole time-consumption (ms): " + std::to_string(timeusage), cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));

    if (writer != nullptr)
      writer->write(img);
    else
    {
      std::string saveimgpath = wdr::path::join({saveroot, imgname + ".jpg"});
      cv::imwrite(saveimgpath, img);
    }
  }

  // 1. 验证后处理没问题
  // 2. 定位改为220，耗时很低的问题，观测forward时候没问题了，存在突然不推理的问题
  // 3. 存在结果随机的问题

  return 0;
}

// 矩形框由左上角模式转为中心点模式
template <typename Dtype>
cv::Rect_<Dtype> cvtRectTL2Center(const cv::Rect_<Dtype> &rect, int offset = 0)
{
  cv::Rect_<Dtype> centerrect;
  centerrect.x = rect.x + (rect.width + offset) / 2;
  centerrect.y = rect.y + (rect.height + offset) / 2;
  centerrect.width = rect.width;
  centerrect.height = rect.height;
  return centerrect;
}

// 矩形框由中心点模式转为左上角模式
template <typename Dtype>
cv::Rect_<Dtype> cvtRectCenter2TL(const cv::Rect_<Dtype> &rect, int offset = 0)
{
  cv::Rect_<Dtype> tlrect;
  tlrect.x = rect.x - (rect.width + offset) / 2;
  tlrect.y = rect.y - (rect.height + offset) / 2;
  tlrect.width = rect.width;
  tlrect.height = rect.height;
  return tlrect;
}

template <typename Dtype>
cv::Rect_<Dtype> estRectangleInCrop(Dtype lsrc, Dtype ldst, const cv::Size_<Dtype> &target)
{
  double scale = double(ldst) / lsrc;
  double w = target.width * scale, h = target.height * scale;

  return cv::Rect_<Dtype>((ldst - w) / 2.0, (ldst - h) / 2.0, w, h);
}

// 获取跟踪窗口
cv::Rect cropTrackWindow(const cv::Mat &img, cv::Mat &trackwindow, const cv::Point2f &tl, const cv::Point2f &br, cv::Vec4i *_pads = nullptr)
{
  int context_xmin = int(tl.x), context_ymin = int(tl.y);
  int context_xmax = int(br.x + 0.5), context_ymax = int(br.y + 0.5);

  // 计算ROI是否在图像内
  int left_pad = std::max(0, -context_xmin),
      top_pad = std::max(0, -context_ymin);
  int right_pad = std::max(0, context_xmax - img.cols),
      bottom_pad = std::max(0, context_ymax - img.rows);

  cv::Rect roi;
  roi.x = context_xmin, roi.y = context_ymin;
  roi.width = context_xmax - context_xmin;
  roi.height = context_ymax - context_ymin;

  if (_pads)
    _pads->val[0] = left_pad, _pads->val[1] = top_pad, _pads->val[2] = right_pad, _pads->val[3] = bottom_pad;

  if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
  {
    img(roi).copyTo(trackwindow);
  }
  else
  {
    cv::Rect imgroi(context_xmin + left_pad, context_ymin + top_pad, context_xmax - context_xmin - left_pad - right_pad, context_ymax - context_ymin - top_pad - bottom_pad);
    cv::Rect dstroi(left_pad, top_pad, imgroi.width, imgroi.height);

    trackwindow.create(top_pad + dstroi.height, left_pad + dstroi.width, CV_MAKETYPE(img.depth(), img.channels()));
    trackwindow.setTo(cv::mean(img));
    img(imgroi).copyTo(trackwindow(dstroi));
  }
}

int track_sequence_capi()
{
  std::string binpath = "projects/torchdnn/data/dcmt/DCMT.bin";
  cv::Rect2f target(653, 220, 55, 40);
  std::string imglistpath = "projects/torchdnn/data/dcmt/ChasingDrones/filelist.txt";
  std::string saveroot = "projects/torchdnn/data/dcmt/ChasingDrones_Results";
  std::string savemode = "video";

  // 1. 提取图像列表
  std::string seqroot = wdr::path::dirname(imglistpath);
  std::vector<std::string> imgnames;
  std::ifstream infile(imglistpath);
  while (!infile.eof())
  {
    std::string tmp;
    infile >> tmp;
    imgnames.push_back(tmp);
    // LOG(INFO) << tmp;
  }

  // 2. 算法初始化
  std::shared_ptr<cv::VideoWriter> writer;

  wdr::DCMTConfigs dcmtcfg;

  // -----------------模型加载部分--------------------
  // 2.1. 加载BIN模型集
  hbPackedDNNHandle_t packed_dnn_handle; // 模型集合指针
  const char *model_file_name = binpath.c_str();
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
  // 2.2. 提取模型集中所有的模型名称
  const char **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
  for (int k = 0; k < model_count; k++) // 输出提取出的所有模型的名称
    LOG(INFO) << "Parsed Model Name: " << std::string(model_name_list[k]);
  // 2.3. 利用目标模型名提取模型指针
  hbDNNHandle_t dnn_handle; // ※模型指针
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // -----------------输入输出内存分配--------------------
  // 2.1. 获取输入/输出Tensor个数
  int input_tensornum = 0, output_tensornum = 0;
  hbDNNGetInputCount(&input_tensornum, dnn_handle);
  hbDNNGetOutputCount(&output_tensornum, dnn_handle);
  LOG(INFO) << "input tensor num: " << input_tensornum << ", output tensor num: " << output_tensornum;

  // 2.2. 获取输入/输出Tensor参数
  std::vector<hbDNNTensorProperties> input_properties, output_properties; // 输入/输出Tensor参数
  input_properties.resize(input_tensornum), output_properties.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
    hbDNNGetInputTensorProperties(&input_properties[k], dnn_handle, k);
  for (int k = 0; k < output_tensornum; k++)
    hbDNNGetOutputTensorProperties(&output_properties[k], dnn_handle, k);

  // 3. 利用参数分配Tensor内存
  std::vector<hbDNNTensor> input_tensors, output_tensors; // ※输入/输出Tensor
  input_tensors.resize(input_tensornum), output_tensors.resize(output_tensornum);
  for (int k = 0; k < input_tensornum; k++)
  {
    const auto &property = input_properties[k];
    input_tensors[k].properties = property;
    input_tensors[k].properties.alignedShape = input_tensors[k].properties.validShape;
    hbSysAllocCachedMem(&input_tensors[k].sysMem[0], property.alignedByteSize);
  }
  for (int k = 0; k < output_tensornum; k++)
  {
    const auto &property = output_properties[k];
    output_tensors[k].properties = property;
    hbSysAllocCachedMem(&output_tensors[k].sysMem[0], property.alignedByteSize);
  }
  LOG(INFO) << "Finish initializing input/output tensors";

  // 3. 开始跟踪
  const int imgnum = imgnames.size();
  bool needinit = true;
  cv::Mat z_bgr, z_box; // 基本跟踪信息
  for (int idximg = 0; idximg < imgnum; idximg++)
  {
    auto &imgname = imgnames[idximg];
    LOG(INFO) << "Processing: " << imgname << "init: " << needinit;

    // 加载图像
    std::string imgpath = wdr::path::join({seqroot, imgname});
    if (!wdr::path::exist(imgpath, true))
      continue;
    cv::Mat img = cv::imread(imgpath);
    if (img.empty())
    {
      LOG(INFO) << "The image is empty, check the valid of : " << imgpath;
      continue;
    }

    if (needinit)
    {
      cv::Rect2f tgt_cen = cvtRectTL2Center(target); // Boxz转中心点模式
      float s_zf = wdr::calTargetLength(tgt_cen.size(), dcmtcfg.context_amount);
      cv::Mat trackwindow;
      cv::Rect2f tgt_tl = cvtRectCenter2TL(tgt_cen);
      auto croproi = cropTrackWindow(img, trackwindow, tgt_tl.tl(), tgt_tl.br());
      LOG(INFO) << "Crop Track Window: " << croproi;
      cv::resize(trackwindow, z_bgr, cv::Size(dcmtcfg.exemplar_size, dcmtcfg.exemplar_size));

      z_box.create(1, 1, CV_MAKETYPE(CV_32F, 4));
      cv::Rect2f _zbbox = estRectangleInCrop<float>(s_zf, dcmtcfg.exemplar_size, tgt_cen.size());
      float *_val = (float *)z_box.data;
      _val[0] = _zbbox.x, _val[1] = _zbbox.y;
      _val[2] = _zbbox.x + _zbbox.width, _val[3] = _zbbox.y + _zbbox.height;
      LOG(INFO) << "z_box point tl: " << _zbbox.tl() << " br: " << _zbbox.br();

      needinit = false;
      continue;
    }
    else
    {
      // 利用当前target估计一个边长
      cv::Rect2f tgt_cen = cvtRectTL2Center(target); // Boxz转中心点模式
      float s_z = wdr::calTargetLength(tgt_cen.size(), dcmtcfg.context_amount);
      float scale_z = dcmtcfg.exemplar_size / s_z;
      float d_search = (float(dcmtcfg.instance_size) - dcmtcfg.exemplar_size) / 2;
      float pad = d_search / scale_z;
      float s_x = s_z + 2 * pad;

      cv::Mat trackwindow, x_crop;
      cv::Rect2f tgt_tl = cvtRectCenter2TL(tgt_cen);
      auto croproi = cropTrackWindow(img, trackwindow, tgt_tl.tl(), tgt_tl.br());
      LOG(INFO) << "Crop Track Window: " << croproi;
      cv::resize(trackwindow, x_crop, cv::Size(dcmtcfg.instance_size, dcmtcfg.instance_size));

      // 上传数据
      {
        auto &tensor = input_tensors[0];
        wdr::BPU::bpuMemcpy(x_crop, tensor, false);
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
      }
      {
        auto &tensor = input_tensors[1];
        wdr::BPU::bpuMemcpy(z_bgr, tensor, false);
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
      }
      {
        auto &tensor = input_tensors[2];
        wdr::BPU::bpuMemcpy(z_box, tensor, false);
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
      }
    }
  }
}