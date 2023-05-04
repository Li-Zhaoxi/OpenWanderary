#include "fcosModelProcess.h"
#include <iostream>
#include <vector>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
// #include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include "method/ptq_fcos_post_process_method.h"
#include "method/method_data.h"
#include "input/input_data.h"

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);

fcosModelProcess::fcosModelProcess(std::string &model_path)
{
  model_file_name = model_path;
}

fcosModelProcess::~fcosModelProcess()
{
  releaseMem();
}

int32_t fcosModelProcess::initModel()
{
  const char* model_name = model_file_name.c_str();
  HB_CHECK_SUCCESS(
      hbDNNInitializeFromFiles(&packed_dnn_handle, &model_name, 1),
      "hbDNNInitializeFromFiles failed");

  const char **model_name_list;
  int model_count = 0;
  HB_CHECK_SUCCESS(
      hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
      "hbDNNGetModelNameList failed");

  HB_CHECK_SUCCESS(
      hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
      "hbDNNGetModelHandle failed");

  output_ptr = std::make_shared<TensorVector>(output_tensors);
  input_ptr = std::make_shared<TensorVector>(input_tensors);
  return prepareTensor();
}

int32_t fcosModelProcess::prepareTensor(){
  int input_count = 0;
  int output_count = 0;
  HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                    "hbDNNGetInputCount failed");
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                    "hbDNNGetInputCount failed");
  input_ptr->tensors.resize(input_count);
  output_ptr->tensors.resize(output_count);
  
  /** Tips:
   * For input memory size:
   * *   input_memSize = input[i].properties.alignedByteSize
   * For output memory size:
   * *   output_memSize = output[i].properties.alignedByteSize
   */
  hbDNNTensor *input = input_ptr->tensors.data();
  for (int i = 0; i < input_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");
    int input_memSize = input[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize),
                     "hbSysAllocCachedMem failed");
    /** Tips:
     * For input tensor, aligned shape should always be equal to the real
     * shape of the user's data. If you are going to set your input data with
     * padding, this step is not necessary.
     * */
    input[i].properties.alignedShape = input[i].properties.validShape;
  }

  hbDNNTensor *output = output_ptr->tensors.data();
  for (int i = 0; i < output_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    int output_memSize = output[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize),
                     "hbSysAllocCachedMem failed");
  }
  return 0;
}

int32_t fcosModelProcess::prepareImage(std::string &image_file)
{
  HB_CHECK_SUCCESS(read_image_2_tensor_as_nv12(image_file),
                   "read_image_2_tensor_as_nv12 failed");
  return 0;
}

int32_t fcosModelProcess::prepareImageBGR(cv::Mat src_mat)
{
  HB_CHECK_SUCCESS(read_image_2_tensor_as_bgr(src_mat),
                   "read_image_2_tensor_as_bgr failed");
  return 0;
}

int32_t fcosModelProcess::prepareImage(cv::Mat src_mat)
{
  HB_CHECK_SUCCESS(read_mat_bgr_2_tensor_as_nv12(src_mat),
                   "read_mat_bgr_2_tensor_as_nv12 failed");
  return 0;
}
int32_t fcosModelProcess::doInfer()
{
  hbDNNTensor *output = output_ptr->tensors.data();
  int input_count = 0;
  int output_count = 0;
  HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                   "hbDNNGetInputCount failed");
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                   "hbDNNGetInputCount failed");
  // make sure memory data is flushed to DDR before inference
  for (int i = 0; i < input_count; i++) {
    hbSysFlushMem(&(input_ptr->tensors[i]).sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }
  hbDNNInferCtrlParam infer_ctrl_param;
  hbDNNTaskHandle_t task_handle = nullptr;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  HB_CHECK_SUCCESS(hbDNNInfer(&task_handle,
                              &output,
                              input_ptr->tensors.data(),
                              dnn_handle,
                              &infer_ctrl_param),
                    "hbDNNInfer failed");
  // wait task done 等待任务结束
  HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0),
                    "hbDNNWaitTaskDone failed");
  // release task handle
  HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");

  // std::cout << "Before prepareOutputData" << std::endl;
  prepareOutputData();
  // std::cout << "After prepareOutputData" << std::endl;

  return 0;
}

PerceptionPtr fcosModelProcess::postProcess(){
  ImageTensor img_tensor;
  img_tensor.tensor = input_ptr->tensors[0];
  img_tensor.ori_image_height = img_height;
  img_tensor.ori_image_width = img_width;
  img_tensor.is_pad_resize = true;
  PTQFcosPostProcessMethod post_process;
  std::string config_str = "\
  {\
    \"top_k\":10,\
    \"score_threshold\":0.6\
  }";
  post_process.InitFromJsonString(config_str);
  PerceptionPtr results;
  results = post_process.DoProcess(&img_tensor, output_ptr);
  /*
  std::cout << "Here output result:" << std::endl
            << "quantiType:" << output_ptr->tensors[0].properties.quantiType << std::endl
            << "tensorLayout:" << output_ptr->tensors[0].properties.tensorLayout << std::endl
            << "type:" << results->type << std::endl
            << "vector Size:" << results->det.size() << std::endl
            << *results.get() << std::endl
            << "Here output result finish" << std::endl;
  std::cout << "post Process finish" << std::endl;*/
  // drawPostImage("0_00.jpg", "test_output.jpg", results);
  return results;
};

int32_t fcosModelProcess::drawPostImage(std::string src_img_file,std::string dst_img_file,PerceptionPtr results)
{
  cv::Mat img_mat = cv::imread(src_img_file, cv::IMREAD_COLOR);
  int i = 0;
  for (auto val1 : results->det)
  {
    float xmin = (val1.bbox.xmin);
    float ymin = val1.bbox.ymin;
    float xmax = val1.bbox.xmax;
    float ymax = val1.bbox.ymax;
    cv::rectangle(img_mat, cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin), (255, 0, 0), 2);
  }
  cv::imwrite(dst_img_file, img_mat);
  return 0;
}
int32_t fcosModelProcess::releaseMem()
{
  
  // release model
  HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle), "hbDNNRelease failed");
  return 0;
}

/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t fcosModelProcess::read_image_2_tensor_as_nv12(std::string &image_file) {
  hbDNNTensor *input_tensor = input_ptr->tensors.data();
  hbDNNTensorProperties Properties = input_tensor->properties;
  int tensor_id = 0;
  int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (bgr_mat.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "image file not exist!";
    return -1;
  }
  // padding and resize
  int max_side = bgr_mat.cols > bgr_mat.rows ? bgr_mat.cols : bgr_mat.rows;
  int min_side = bgr_mat.cols < bgr_mat.rows ? bgr_mat.cols : bgr_mat.rows;
  this->img_width = bgr_mat.cols;
  this->img_height = bgr_mat.rows;
  // std::cout << "max_side: " << max_side << std::endl;
  cv::Mat mat_input;
  mat_input.create(max_side, max_side, bgr_mat.type());
  cv::copyMakeBorder(bgr_mat, mat_input, 0, max_side - min_side, 0, 0, cv::BORDER_CONSTANT);
 
  cv::Mat mat;
  cv::resize(mat_input, mat, cv::Size(512, 512), 0, 0);
  cv::imwrite("temp_output_img512X512.jpg", mat);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
    VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

  // copy y data
  auto data = input_tensor->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);
  // for (int i = 0; i < 10;++i)
  // {
  //   std::cout << (int)((uint8_t*)data)[i] << std::endl;
  // }
  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
  uint8_t *u_data = nv12_data + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++)
  {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

int32_t fcosModelProcess::read_mat_bgr_2_tensor_as_nv12(cv::Mat src_mat) {
  hbDNNTensor *input_tensor = input_ptr->tensors.data();
  hbDNNTensorProperties Properties = input_tensor->properties;
  int tensor_id = 0;
  int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  // cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (src_mat.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "src_mat is empty!";
    return -1;
  }
  // padding and resize
  int max_side = src_mat.cols > src_mat.rows ? src_mat.cols : src_mat.rows;
  int min_side = src_mat.cols < src_mat.rows ? src_mat.cols : src_mat.rows;
  this->img_width = src_mat.cols;
  this->img_height = src_mat.rows;
  // std::cout << "max_side: " << max_side << std::endl;
  cv::Mat mat_input;
  mat_input.create(max_side, max_side, src_mat.type());
  cv::copyMakeBorder(src_mat, mat_input, 0, max_side - min_side, 0, 0, cv::BORDER_CONSTANT);

  cv::Mat mat;
  cv::resize(mat_input, mat, cv::Size(512, 512), 0, 0);
  // cv::imwrite("temp_output_img512X512.jpg", mat);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
    VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

  // copy y data
  auto data = input_tensor->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);
  // for (int i = 0; i < 10;++i)
  // {
  //   std::cout << (int)((uint8_t*)data)[i] << std::endl;
  // }
  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
  uint8_t *u_data = nv12_data + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++)
  {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}


int fcosModelProcess::postProcess(std::vector<int> &idx, std::vector<cv::Vec4f> &detbbox, std::vector<float> &score)
{
  PerceptionPtr dets = postProcess();
  idx.clear();
  detbbox.clear();
  score.clear();
  idx.resize(dets->det.size());
  detbbox.resize(dets->det.size());
  score.resize(dets->det.size());
  for (int i = 0; i < dets->det.size(); ++i)//Detection val : dets->det)
  {
    idx[i] = (dets->det[i].id);
    detbbox[i] = (cv::Vec4f(dets->det[i].bbox.xmin, dets->det[i].bbox.ymin,
                            dets->det[i].bbox.xmax, dets->det[i].bbox.ymax));
    score[i] = (dets->det[i].score);
  }
  return dets->det.size();
}


void drawPoseImage(cv::Mat &imgC, std::vector<int> &idx, std::vector<cv::Vec4f> &detbbox, std::vector<cv::Scalar> &colors, int thickness)
{
  if (imgC.channels() == 1)
		cvtColor(imgC, imgC, cv::COLOR_GRAY2BGR);
  int bboxnum = detbbox.size();
  int colornum = colors.size();
  for (int i = 0; i < bboxnum; i++)
  {
    float xmin = detbbox[i][0];
    float ymin = detbbox[i][1];
    float xmax = detbbox[i][2];
    float ymax = detbbox[i][3];
    int dst_class = idx[i];
    if (dst_class < colornum)
      cv::rectangle(imgC, cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin), colors[dst_class], thickness);
  }
}

void drawPostImage(unsigned char *_imgC, int rows, int cols, int bboxnum, signed int *_idx, float *_detbbox, signed int *_colors, int colornum, int thickness)
{
  cv::Mat imgC(rows, cols, CV_8UC3, _imgC);
  std::vector<int> idx(bboxnum);
  std::vector<cv::Vec4f> detbbox(bboxnum);
  for(int i = 0; i < bboxnum; i++)
  {
    idx[i] = _idx[i];
    detbbox[i][0] = _detbbox[i * 4 + 0];
    detbbox[i][1] = _detbbox[i * 4 + 1];
    detbbox[i][2] = _detbbox[i * 4 + 2];
    detbbox[i][3] = _detbbox[i * 4 + 3];
  }
  std::vector<cv::Scalar> colors(colornum);
  for(int i = 0; i < colornum; i++)
  {
    colors[i] = cv::Scalar(_colors[i * 3], _colors[i * 3 + 1], _colors[i * 3 + 2]);
  }
  drawPoseImage(imgC, idx, detbbox, colors, thickness);

  if (imgC.data != _imgC)
	{
		memcpy(_imgC, imgC.data, sizeof(unsigned char) * 3 * rows * cols);
	}

}






void fcosModelProcess::prepareOutputData()
{
  // 组织output_tensors中的输出数据
  hbDNNTensor *output = output_ptr->tensors.data();
  int output_count = 0;
  hbDNNGetOutputCount(&output_count, dnn_handle);
  // std::cout << "output_count: " << output_count << std::endl;
  outputs_data.clear();
  output_data_size.clear();
  output_data_shap.resize(output_count, std::vector<int32_t>(8, 0));
  outputs_data.resize(output_count);
  output_data_size.resize(output_count);
  for (int i = 0; i < output_count; ++i)
  {
    // std::vector<int> temp_vec;
    output_data_shap[i].resize(output[i].properties.validShape.numDimensions);
    for (int j = 0; j < output[i].properties.validShape.numDimensions;++j)
    {
      output_data_shap[i][j] = output[i].properties.validShape.dimensionSize[j];
      // temp_vec.push_back(output[i].properties.validShape.dimensionSize[j]);
    }
    outputs_data[i] = output[i].sysMem[0].virAddr;
    output_data_size[i] = output[i].sysMem[0].memSize;
    // output_data_shap.push_back(temp_vec);
  }

}




// 使用demo
//   std::vector<void *> *outputs = fmp.getOutputsData();
//   std::vector<int32_t> *output_size = fmp.getOutputsDataSize();
//   std::vector<std::vector<int32_t>> *output_shap = fmp.getOutputsDataShap();
//   for (int i = 0; i < outputs->size(); ++i)
//   {
//     float *data = (float *)(*outputs)[i];
//     for (int j = 0; j < 10; ++j)
//     {
//       std::cout << ", " << data[j];
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
//   for (int i = 0; i < output_size->size(); ++i)
//   {
//     std::cout << (*output_size)[i] << std::endl;
//   }
//   std::cout << std::endl;
//   for (int i = 0; i < (*output_shap).size(); ++i)
//   {
//     std::vector<int32_t> temp_v = (*output_shap)[i];
//     for (int j = 0; j < temp_v.size(); ++j)
//     {
//       std::cout << ". " << temp_v[j];
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;

int32_t fcosModelProcess::read_image_2_tensor_as_bgr(std::string &image_file) {
  hbDNNTensor *input_tensor = input_ptr->tensors.data();
  hbDNNTensorProperties Properties = input_tensor->properties;
  int tensor_id = 0;
  int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (bgr_mat.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "image file not exist!";
    return -1;
  }
  // padding and resize
  double scale = 512.0 / bgr_mat.rows;
  cv::Mat mat;
  cv::resize(bgr_mat, mat, cv::Size(), scale, scale);

  this->img_width = bgr_mat.cols;
  this->img_height = bgr_mat.rows;
  // std::cout << "max_side: " << max_side << std::endl;
  cv::Mat mat_pad;
  // mat_input.create(max_side, max_side, bgr_mat.type());
  cv::copyMakeBorder(mat, mat_pad, 0, 0, 0, input_w - mat.cols, cv::BORDER_CONSTANT);
  // std::cout << "mat_pad_cols: " << mat_pad.cols << " , mat_pad_rows: " << mat_pad.rows << std::endl;

  // cv::imwrite("temp_output_img512X512.jpg", mat);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
    VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    return -1;
  }

  // copy bgr data to input_tensor
  auto data = input_tensor->sysMem[0].virAddr;
  int32_t _size = input_h * input_w * 3;
  memcpy(reinterpret_cast<uint8_t *>(data), mat_pad.data, _size);
  return 0;
}

int32_t fcosModelProcess::read_image_2_tensor_as_bgr(cv::Mat src_mat) {
  hbDNNTensor *input_tensor = input_ptr->tensors.data();
  hbDNNTensorProperties Properties = input_tensor->properties;
  int tensor_id = 0;
  int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }
  // std::cout << "input_h: " << input_h << " , input_w: " << input_w << std::endl;
  // cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (src_mat.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "src_mat is empty!";
    return -1;
  }
  // padding and resize
  double scale = 512.0 / src_mat.rows;
  cv::Mat mat;
  cv::resize(src_mat, mat, cv::Size(), scale, scale);
  // std::cout << "src_mat_cols: " << src_mat.cols << " , src_mat_rows: " << src_mat.rows << std::endl;
  // std::cout << "mat_cols: " << mat.cols << " , mat_rows: " << mat.rows << std::endl;


  this->img_width = src_mat.cols;
  this->img_height = src_mat.rows;
  // std::cout << "max_side: " << max_side << std::endl;
  cv::Mat mat_pad;
  // mat_pad.create(max_side, max_side, src_mat.type());
  cv::copyMakeBorder(mat, mat_pad, 0, 0, 0, input_w - mat.cols, cv::BORDER_CONSTANT);
  // std::cout << "mat_pad_cols: " << mat_pad.cols << " , mat_pad_rows: " << mat_pad.rows << std::endl;
    cv::imwrite("temp_output_img512X512.jpg", mat_pad);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
    VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    return -1;
  }

    // copy bgr data to input_tensor
  auto data = input_tensor->sysMem[0].virAddr;
  int32_t _size = input_h * input_w * 3;
  memcpy(reinterpret_cast<uint8_t *>(data), mat_pad.data, _size);

  // cv::Mat yuv_mat;
  // cv::cvtColor(mat_pad, yuv_mat, cv::COLOR_BGR2YUV_I420);
  // uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

  // // copy y data
  // auto data = input_tensor->sysMem[0].virAddr;
  // int32_t y_size = input_h * input_w;
  // memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);
  // // for (int i = 0; i < 10;++i)
  // // {
  // //   std::cout << (int)((uint8_t*)data)[i] << std::endl;
  // // }
  // // copy uv data
  // int32_t uv_height = input_h / 2;
  // int32_t uv_width = input_w / 2;
  // uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
  // uint8_t *u_data = nv12_data + y_size;
  // uint8_t *v_data = u_data + uv_height * uv_width;

  // for (int32_t i = 0; i < uv_width * uv_height; i++)
  // {
  //   *nv12++ = *u_data++;
  //   *nv12++ = *v_data++;
  // }
  return 0;
}