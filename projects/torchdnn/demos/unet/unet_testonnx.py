import numpy as np
import cv2
import scipy 
import onnxruntime

def preprocess(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = img.transpose(2, 0, 1) # 通道由HWC变为CHW
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1CHW
  
  # 图像归一化操作
  img = img.astype("float32")
  mu = np.array([123.675,116.28,103.53], dtype=np.float32)
  s= np.array([0.01712475,0.017507,0.01742919], dtype=np.float32)
  for c in range(img.shape[1]):
    img[:, c, :, :] = (img[:, c, :, :] - mu[c]) * s[c]
  return img

def postprocess(outputs: np.ndarray) -> np.ndarray:
  # 元素归一化到[0,1]之后，选择前景部分的数据
  y_list = scipy.special.softmax(outputs, axis = 1)[:, 1, :, :] 
  # 大于0.5的就是前景
  y_list = (y_list > 0.5).astype(np.uint8) * 255 
  return y_list


imgpath = "D:/01 - datasets/008 - 2DMRACerebrovascular/images/mra_img_12.jpg"
onnxpath = "unet.onnx"

img = cv2.imread(imgpath)
if len(img.shape) == 2:
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 4 # 设置线程数
session = onnxruntime.InferenceSession(onnxpath, sess_options = sess_options)

datain = preprocess(img, 256, 256)
inputs = {session.get_inputs()[0].name: datain}
outputs = session.run(None, inputs)
pred = postprocess(outputs[0])[0]

cv2.imwrite("pred.png", pred)