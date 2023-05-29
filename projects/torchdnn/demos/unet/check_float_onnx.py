### 注意：改代码仅能在OE docker中运行
import numpy as np
import cv2
import os
from prepare_functions import get_rgb_image, postprocess
# horizon_nn 是在docker中才有的包
from horizon_nn import horizon_onnxruntime
from horizon_nn import horizon_onnx

# 板端数据预处理函数, 注意这里用的是input_layout_train排布，input_type_rt类型
def preprocess_floatmodel(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = img.transpose(2, 0, 1) # 通道由HWC变为CHW
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1CHW
  img = np.ascontiguousarray(img) # 板端的推理是封装的C++，安全起见这里约束矩阵内存连续
  return img

dataroot = "/data/horizon_x3/data"
imgpath = os.path.join(dataroot, "mra_img_12.jpg")
# onnxpath = os.path.join(dataroot, "model_output", "unet_original_float_model.onnx")
onnxpath = os.path.join(dataroot, "model_output", "unet_optimized_float_model.onnx")

# 加载图像和ONNX模型
img = get_rgb_image(imgpath)
session = horizon_onnxruntime.InferenceSession(onnxpath)

# 校验预处理函数
datain = preprocess_floatmodel(img, 256, 256) # 1x256x256x3
# 构建输入并推理，记得要转为float32
inputs = {session.get_inputs()[0].name: (datain.astype(np.int32) - 128).astype(np.float32)}
outputs = session.run(None, inputs)

# 后处理并保存结果
pred = postprocess(outputs[0])[0]
# cv2.imwrite(os.path.join(dataroot, "pred_original_onnx.png"), pred)
cv2.imwrite(os.path.join(dataroot, "pred_optimized_onnx.png"), pred)