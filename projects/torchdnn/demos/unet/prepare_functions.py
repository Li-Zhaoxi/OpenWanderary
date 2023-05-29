import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

import numpy as np
import cv2
from scipy.special import softmax

def get_rgb_image(imgpath: str) -> np.ndarray:
  img = cv2.imread(imgpath)
  if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  else: 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def get_bgr_image(imgpath: str) -> np.ndarray:
  img = cv2.imread(imgpath)
  if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img

def check_matrix_equal(src: np.ndarray, dst: np.ndarray, thre, saveroot, name):
  assert isinstance(src, np.ndarray), f"src must be np.ndarray, but it is {type(src)}"
  assert isinstance(dst, np.ndarray), f"dst must be np.ndarray, but it is {type(dst)}"
  
  assert len(src.shape) in [2, 3, 4] and src.shape == dst.shape, f"the length of the shape must be 4 and the shapes must be equal. src: {src.shape}, dst: {dst.shape}"
  
  if len(src.shape) == 4:
    for idxb in range(src.shape[0]):
      for idxc in range(src.shape[1]):
        diff = np.abs(src[idxb, idxc, ...] - dst[idxb, idxc, ...])
        if np.max(diff) < thre:
          continue
        imgerr = (diff >= thre).astype(np.uint8) * 255
        print(f"Discovered an invalid matrix at (b:{idxb}, c:{idxc}), max diff: {np.max(diff)}")
        cv2.imwrite(os.path.join(saveroot, f"err_{name}_{idxb}_{idxc}.png"), imgerr)
  elif len(src.shape) == 3:
    for idxb in range(src.shape[0]):
      diff = np.abs(src[idxb, ...] - dst[idxb, ...])
      if np.max(diff) < thre:
        continue
      imgerr = (diff >= thre).astype(np.uint8) * 255
      print(f"Discovered an invalid matrix at (b:{idxb}), max diff: {np.max(diff)}")
      cv2.imwrite(os.path.join(saveroot, f"err_{name}_{idxb}.png"), imgerr)
  elif len(src.shape) == 2:
    diff = np.abs(src - dst)
    if np.max(diff) >= thre:
      imgerr = (diff >= thre).astype(np.uint8) * 255
      print(f"Discovered an invalid matrix, max diff: {np.max(diff)}")
      cv2.imwrite(os.path.join(saveroot, f"err_{name}.png"), imgerr)

  print(f"finish the check task: {name}")


# 无torch依赖的预处理函数: img为RGB通道，排布HWC
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

# 无torch依赖的后处理函数
def postprocess(outputs: np.ndarray) -> np.ndarray:
  # 元素归一化到[0,1]之后，选择前景部分的数据
  y_list = softmax(outputs, axis = 1)[:, 1, :, :] 
  # 大于0.5的就是前景
  y_list = (y_list > 0.5).astype(np.uint8) * 255 
  return y_list


# 校验数据预处理函数, 注意输入的img是RGB通道
def preprocess_calibration(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = img.transpose(2, 0, 1) # 通道由HWC变为CHW
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1CHW
  return img

# 板端数据预处理函数, 注意输入的img是bgr通道
def preprocess_onboard(img: np.ndarray, modelh, modelw) -> np.ndarray:
  img = cv2.resize(img, (modelw, modelh))# Resize图像尺寸
  img = np.expand_dims(img, 0) # 增加一维，此时维度为1HWC
  img = np.ascontiguousarray(img) # 板端的推理是封装的C++，安全起见这里约束矩阵内存连续
  return img