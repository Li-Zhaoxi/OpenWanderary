
import numpy as np
import cv2
import os
from hobot_vio import libsrcampy as srcampy


saveroot = "examples/MIPI/data/images"
if not os.path.exists(saveroot):
  os.system(f"mkdir -p {saveroot}")

cam = srcampy.Camera() 
# 打开CAM2的相机，并保存
ret = cam.open_cam(0, 2, 30, 1920, 1080)
print(ret)
if ret != 0:
  exit()

for idx in range(20):
  print(idx)
  origin_image = cam.get_img(1, 1920, 1080)
  origin_nv12 = np.frombuffer(origin_image, dtype=np.uint8).reshape(1620, 1920)
  origin_bgr = cv2.cvtColor(origin_nv12, cv2.COLOR_YUV420SP2RGB) 
  cv2.imwrite(f"{saveroot}/cam_{idx}.jpg", origin_bgr)

cam.close_cam()