import os
import numpy as np
import cv2
import scipy 
import onnxruntime
from prepare_functions import get_rgb_image, preprocess, postprocess

dataroot = "data/unet"
imgpath = os.path.join(dataroot, "mra_img_12.jpg")
onnxpath = os.path.join(dataroot, "unet.onnx")

# 加载图像和ONNX模型
img = get_rgb_image(imgpath) # 获取RGB的图像
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 4 # 设置线程数
session = onnxruntime.InferenceSession(onnxpath, sess_options = sess_options)

# ONNX推理
datain = preprocess(img, 256, 256)
inputs = {session.get_inputs()[0].name: datain} # 构建onnx输入，是个dict
outputs = session.run(None, inputs)
# outputs是个列表，记录了模型的所有输出，unet输出只有一个所以选择[0]
pred = postprocess(outputs[0])[0] 
for j in range(pred.shape[0]):
  cv2.imwrite(os.path.join(dataroot, f"pred_onnx_b{j}.png"), pred[j].astype(np.uint8))
