from wanderary.visualization import ImageShow
import numpy as np
import cv2
from hobot_vio import libsrcampy as srcampy


im_show = ImageShow() # 可视化工具初始化

cam_raspi = srcampy.Camera()
res_raspi = cam_raspi.open_cam(0, 2, 30, 1920, 1080) # 设置树莓派相机参数

cam_x3pi = srcampy.Camera()
res_x3pi = cam_x3pi.open_cam(1, 1, 30, 1920, 1080) # 设置X3派相机参数

if res_raspi != 0 or res_x3pi != 0:
  raise(Exception(f"Camera Open Failed. res_raspi: {res_raspi}, res_x3pi: {res_x3pi}"))
else:
  print("All Cameras are open.")
  
for idx in range(20):
  print(idx)
  origin_image_raspi = cam_raspi.get_img(2, 1920, 1080) # 获取相机数据流
  origin_nv12_raspi = np.frombuffer(origin_image_raspi, dtype=np.uint8).reshape(1620, 1920)
  origin_bgr_raspi = cv2.cvtColor(origin_nv12_raspi, cv2.COLOR_YUV420SP2RGB) 
  cv2.putText(origin_bgr_raspi, "Camera CAM2 [RasPi]", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)  
  
  origin_image_x3pi = cam_x3pi.get_img(2, 1920, 1080) # 获取相机数据流
  origin_nv12_x3pi = np.frombuffer(origin_image_x3pi, dtype=np.uint8).reshape(1620, 1920)
  origin_bgr_x3pi = cv2.cvtColor(origin_nv12_x3pi, cv2.COLOR_YUV420SP2RGB) 
  cv2.putText(origin_bgr_x3pi, "Camera CAM1 [X3Pi]", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)  
  
  imgshow = cv2.hconcat([origin_bgr_raspi, origin_bgr_x3pi])
  im_show.show(imgshow)

im_show.close()
cam_raspi.close_cam()
cam_x3pi.close_cam()
