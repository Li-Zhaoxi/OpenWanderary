import numpy as np
import cv2
import scipy 
import onnxruntime

def preprocess_onboard(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1CHW
  img = np.ascontiguousarray(img) # 板端的推理是封装的C++，安全起见这里约束矩阵内存连续
  return img


def postprocess(outputs: np.ndarray) -> np.ndarray:
  # 元素归一化到[0,1]之后，选择前景部分的数据
  y_list = scipy.special.softmax(outputs, axis = 1)[:, 1, :, :] 
  # 大于0.5的就是前景
  y_list = (y_list > 0.5).astype(np.uint8) * 255 
  return y_list


imgpath = "D:/01 - datasets/008 - 2DMRACerebrovascular/images/mra_img_12.jpg"
onnxpath = "D:/05 - 项目/01 - 旭日x3派/BPUCodes/unet_onnx2bin/model_output/unet_quantized_model.onnx"

img = cv2.imread(imgpath)
if len(img.shape) == 2:
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 4 # 设置线程数
session = onnxruntime.InferenceSession(onnxpath, sess_options = sess_options)

datain = preprocess_onboard(img, 256, 256)
inputs = {session.get_inputs()[0].name: datain}
outputs = session.run(None, inputs)
print(len(outputs), outputs[0].shape)
# pred = postprocess(outputs[0])[0]

# cv2.imwrite("pred.png", pred)