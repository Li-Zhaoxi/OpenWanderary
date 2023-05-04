#include <iostream>
#include "FCOS/fcosModelProcess.h"
#include "FCOS/method/method_data.h"

int main()
{
  std::string model_file = "examples/modules/fcos_dog_v6.bin";
  std::string src_img_file= "examples/modules/0_00.jpg";
  std::string dst_img_file= "0_00_output.jpg";
  std::string dst_img_file2= "0_00_output2.jpg";

  fcosModelProcess fmp(model_file);
  HB_CHECK_SUCCESS(fmp.initModel(),"fmpInit failed");
  // int32_t img_wid, img_hgt;
  cv::Mat src = cv::imread(src_img_file.c_str(), cv::IMREAD_COLOR);
  HB_CHECK_SUCCESS(fmp.prepareImageBGR(src), "fmpPrepareImg failed");
  HB_CHECK_SUCCESS(fmp.doInfer(), "fmpDoInfer failed");
  PerceptionPtr ret = fmp.postProcess();
  fmp.drawPostImage(src_img_file, dst_img_file, ret);

  // fmp.prepareImage(src);
  // fmp.doInfer();
  // PerceptionPtr ret2 = fmp.postProcess();
  // fmp.drawPostImage(src_img_file, dst_img_file2, ret2);

  return 0;
}

// #include <iostream>
// #include <vector>

// #include "dnn/hb_dnn.h"
// #include "dnn/hb_sys.h"
// #include "gflags/gflags.h"
// #include "glog/logging.h"
// #include "opencv2/imgproc.hpp"
// #include <opencv2/imgcodecs.hpp>

// #include "method/ptq_fcos_post_process_method.h"
// #include "method/method_data.h"
// #include "input/input_data.h"

// #define HB_CHECK_SUCCESS(value, errmsg)                              \
//   do {                                                               \
//     /*value can be call of function*/                                \
//     auto ret_code = value;                                           \
//     if (ret_code != 0) {                                             \
//       VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
//       return ret_code;                                               \
//     }                                                                \
//   } while (0);

// int prepare_tensor(hbDNNTensor *input_tensor,
//                    hbDNNTensor *output_tensor,
//                    hbDNNHandle_t dnn_handle) {
//   int input_count = 0;
//   int output_count = 0;
//   hbDNNGetInputCount(&input_count, dnn_handle);
//   hbDNNGetOutputCount(&output_count, dnn_handle);

//   /** Tips:
//    * For input memory size:
//    * *   input_memSize = input[i].properties.alignedByteSize
//    * For output memory size:
//    * *   output_memSize = output[i].properties.alignedByteSize
//    */
//   hbDNNTensor *input = input_tensor;
//   for (int i = 0; i < input_count; i++) {
//     HB_CHECK_SUCCESS(
//         hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i),
//         "hbDNNGetInputTensorProperties failed");
//     int input_memSize = input[i].properties.alignedByteSize;
//     HB_CHECK_SUCCESS(hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize),
//                      "hbSysAllocCachedMem failed");
//     /** Tips:
//      * For input tensor, aligned shape should always be equal to the real
//      * shape of the user's data. If you are going to set your input data with
//      * padding, this step is not necessary.
//      * */
//     input[i].properties.alignedShape = input[i].properties.validShape;
//   }

//   hbDNNTensor *output = output_tensor;
//   for (int i = 0; i < output_count; i++) {
//     HB_CHECK_SUCCESS(
//         hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i),
//         "hbDNNGetOutputTensorProperties failed");
//     int output_memSize = output[i].properties.alignedByteSize;
//     HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize),
//                      "hbSysAllocCachedMem failed");
//   }
//   return 0;
// }

// /** You can define read_image_2_tensor_as_other_type to prepare your data **/
// int32_t read_image_2_tensor_as_nv12(std::string &image_file,
//                                     hbDNNTensor *input_tensor,
//                                     int32_t &img_width,
//                                     int32_t &img_height) {
//   hbDNNTensorProperties Properties = input_tensor->properties;
//   int tensor_id = 0;
//   int input_h = Properties.validShape.dimensionSize[1];
//   int input_w = Properties.validShape.dimensionSize[2];
//   if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
//     input_h = Properties.validShape.dimensionSize[2];
//     input_w = Properties.validShape.dimensionSize[3];
//   }

//   cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
//   if (bgr_mat.empty()) {
//     VLOG(EXAMPLE_SYSTEM) << "image file not exist!";
//     return -1;
//   }
//   // padding and resize
//   int max_side = bgr_mat.cols > bgr_mat.rows ? bgr_mat.cols : bgr_mat.rows;
//   int min_side = bgr_mat.cols < bgr_mat.rows ? bgr_mat.cols : bgr_mat.rows;
//   img_width = bgr_mat.cols;
//   img_height = bgr_mat.rows;
//   // std::cout << "max_side: " << max_side << std::endl;
//   cv::Mat mat_input;
//   mat_input.create(max_side, max_side, bgr_mat.type());
//   cv::copyMakeBorder(bgr_mat, mat_input, 0, max_side - min_side, 0, 0, cv::BORDER_CONSTANT);
 
//   cv::Mat mat;
//   cv::resize(mat_input, mat, cv::Size(512, 512), 0, 0);
//   // cv::imwrite("temp_output_img512X512.jpg", mat);
//   // convert to YUV420
//   if (input_h % 2 || input_w % 2) {
//     VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
//     return -1;
//   }
//   cv::Mat yuv_mat;
//   cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
//   uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

//   // copy y data
//   auto data = input_tensor->sysMem[0].virAddr;
//   int32_t y_size = input_h * input_w;
//   memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);
// // for (int i = 0; i < 10;++i)
// //   {
// //     std::cout << (int)((uint8_t*)data)[i] << std::endl;
// //   }
//   // copy uv data
//   int32_t uv_height = input_h / 2;
//   int32_t uv_width = input_w / 2;
//   uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
//   uint8_t *u_data = nv12_data + y_size;
//   uint8_t *v_data = u_data + uv_height * uv_width;

//   for (int32_t i = 0; i < uv_width * uv_height; i++)
//   {
//     *nv12++ = *u_data++;
//     *nv12++ = *v_data++;
//   }
//   return 0;
// }

// int main() {
//     {
//     std::string model_file = "./fcos_nv12_to_yuv444_preprocess.bin";
//     std::string src_img_file= "0_00.jpg";
//     std::string dst_img_file= "0_00_output.jpg";
//     std::string dst_img_file2= "0_00_output2.jpg";
//     fcosModelProcess fmp(model_file);
//     HB_CHECK_SUCCESS(fmp.initModel(),"fmpInit failed");
//     int32_t img_wid, img_hgt;
    
//     HB_CHECK_SUCCESS(fmp.prepareImage(src_img_file, img_wid, img_hgt),"fmpPrepareImg failed");
//     std::cout << "img_wid: " << img_wid << " img_hgt: " << img_hgt << std::endl;
//     HB_CHECK_SUCCESS(fmp.doInfer(), "fmpDoInfer failed");
//     PerceptionPtr ret = fmp.postProcess();
//     fmp.drawPostImage(src_img_file, dst_img_file, ret);

//   }
//   // std::cout << "model process begin!" << std::endl;
//   std::string image_file = "0_00.jpg";
//   // 第一步加载模型
//   hbPackedDNNHandle_t packed_dnn_handle;
//   const char* model_file_name= "./fcos_nv12_to_yuv444_preprocess.bin";
//   HB_CHECK_SUCCESS(
//       hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
//       "hbDNNInitializeFromFiles failed");
//   std::cout << "hbDNNInitializeFromFiles finish" << std::endl;

//   // 第二步获取模型名称
//   const char **model_name_list;
//   int model_count = 0;
//   HB_CHECK_SUCCESS(
//       hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
//       "hbDNNGetModelNameList failed");
//   std::cout << "hbDNNGetModelNameList finish" << std::endl;

//   // 第三步获取dnn_handle
//   hbDNNHandle_t dnn_handle;
//   HB_CHECK_SUCCESS(
//       hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
//       "hbDNNGetModelHandle failed");
//   std::cout << "hbDNNGetModelHandle finish" << std::endl;

//   // 第四步准备输入、输出的数据空间
//   TensorVector output_tensors;
//   TensorVector input_tensors;
//   std::shared_ptr<TensorVector> output_ptr = std::make_shared<TensorVector>(output_tensors);
//   std::shared_ptr<TensorVector> input_ptr = std::make_shared<TensorVector>(input_tensors);
//   int input_count = 0;
//   int output_count = 0;
//   HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
//                     "hbDNNGetInputCount failed");
//   HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
//                     "hbDNNGetInputCount failed");
//   input_ptr->tensors.resize(input_count);
//   output_ptr->tensors.resize(output_count);
//   prepare_tensor(input_ptr->tensors.data(), output_ptr->tensors.data(), dnn_handle);
//   std::cout << "prepare_tensor finish" << std::endl;

//   // 第五步填充输入数据
//   int32_t img_width, img_height;
//   HB_CHECK_SUCCESS(
//       read_image_2_tensor_as_nv12(image_file, &(input_ptr->tensors[0]), img_width, img_height),
//       "read_image_2_tensor_as_nv12 failed");
//   std::cout << "read_image_2_tensor_as_nv12 finish" << std::endl;

//   // 第六步推理模型
//   hbDNNTaskHandle_t task_handle = nullptr;
//   hbDNNTensor *output = output_ptr->tensors.data();
//   // make sure memory data is flushed to DDR before inference
//   for (int i = 0; i < input_count; i++) {
//     hbSysFlushMem(&(input_ptr->tensors[i]).sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
//   }
//   hbDNNInferCtrlParam infer_ctrl_param;
//   HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
//   HB_CHECK_SUCCESS(hbDNNInfer(&task_handle,
//                               &output,
//                               input_ptr->tensors.data(),
//                               dnn_handle,
//                               &infer_ctrl_param),
//                     "hbDNNInfer failed");
//   // wait task done 等待任务结束
//   HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0),
//                     "hbDNNWaitTaskDone failed");
//   std::cout << "hbDNNWaitTaskDone finish" << std::endl;

//   //第七步后处理
//   ImageTensor img_tensor;
//   img_tensor.tensor = input_ptr->tensors[0];
//   img_tensor.ori_image_height = img_height;
//   img_tensor.ori_image_width = img_width;
//   img_tensor.is_pad_resize = true;
//   PTQFcosPostProcessMethod post_process;
//   std::string config_str = "\
//   {\
//     \"top_k\":5,\
//     \"score_threshold\":0.6\
//   }";
//   post_process.InitFromJsonString(config_str);
//   PerceptionPtr results;
//   results = post_process.DoProcess(&img_tensor, output_ptr);
//   std::cout << "Here output result:" << std::endl
//             << "quantiType:" << output_ptr->tensors[0].properties.quantiType << std::endl
//             << "tensorLayout:" << output_ptr->tensors[0].properties.tensorLayout << std::endl
//             << "type:" << results->type << std::endl
//             << "vector Size:" << results->det.size() << std::endl
//             << *results.get() << std::endl
//             << "Here output result finish" << std::endl;
//   std::cout << "post Process finish" << std::endl;

//   // 第八步输出检测结果图
//   cv::Mat img_mat = cv::imread(image_file, cv::IMREAD_COLOR);
//   int i = 0;
//   for (auto val1 : results->det)
//   {
//     float xmin = (val1.bbox.xmin);
//     float ymin = val1.bbox.ymin;
//     float xmax = val1.bbox.xmax;
//     float ymax = val1.bbox.ymax;
//     cv::rectangle(img_mat, cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin), (0, 255, 0), 2);
//   }
//   cv::imwrite("detect_box_output.jpg", img_mat);
  
//   // 第九步释放内存
//   // release task handle
//   HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
//   // // free input mem
//   // for (int i = 0; i < input_count; i++) {
//   //   HB_CHECK_SUCCESS(hbSysFreeMem(&(input_ptr->tensors[i].sysMem[0])),
//   //                    "hbSysFreeMem failed");
//   // }
//   // // free output mem
//   // for (int i = 0; i < output_count; i++) {
//   //   HB_CHECK_SUCCESS(hbSysFreeMem(&(output_ptr->tensors[i].sysMem[0])),
//   //                    "hbSysFreeMem failed");
//   // }
//   // release model
//   HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle), "hbDNNRelease failed");

// 	printf("Hello World this is end \n");


//   return 0;
// }

