import numpy as np
import cv2
import os

def preprocess_calibration(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = img.transpose(2, 0, 1) # 通道由HWC变为CHW
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1CHW
  return img

imgroot = "D:/01 - datasets/008 - 2DMRACerebrovascular/images"
calibroot = "D:/01 - datasets/008 - 2DMRACerebrovascular/Calibration"

imgroot = "D:/01 - datasets/008 - 2DMRACerebrovascular/images"
calibroot = "D:/01 - datasets/008 - 2DMRACerebrovascular/Calibration"
imgcount, total = 0, 50
for imgname in os.listdir(imgroot):
  if not imgname.endswith(".jpg"):
    continue
  img = cv2.imread(os.path.join(imgroot, imgname))
  if len(img.shape) == 2: # 约束图像为RGB通道
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  calibdata = preprocess_calibration(img, 256, 256) # 校验数据预处理函数
  calibdata.astype(np.uint8).tofile(os.path.join(calibroot, imgname + ".rgbchw"))
  imgcount += 1
  if imgcount >= total:
    break





