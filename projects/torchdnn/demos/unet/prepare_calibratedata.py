import numpy as np
import cv2
import os
from prepare_functions import get_rgb_image, preprocess_calibration

dataroot = "data/unet"
imgroot = os.path.join(dataroot, "images")
calibroot = os.path.join(dataroot, "calibration")
for imgname in os.listdir(imgroot):
  img = get_rgb_image(os.path.join(imgroot, imgname))
  calibdata = preprocess_calibration(img, 256, 256) # 校验数据预处理函数
  calibdata.astype(np.uint8).tofile(os.path.join(calibroot, imgname + ".rgbchw"))





