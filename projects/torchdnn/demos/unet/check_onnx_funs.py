import numpy as np
import onnxruntime
import cv2
import scipy 
import os
from prepare_functions import check_matrix_equal, preprocess, postprocess

dataroot = "data/unet"
data = np.load(os.path.join(dataroot, "unet_checkstage1.npz"))

image = data["image"]
datain = data["datain"]
dataout = data["dataout"]
pred = data["pred"]

###### 检查ONNX有效性
onnxpath = os.path.join(dataroot, "unet.onnx")
session = onnxruntime.InferenceSession(onnxpath)
inputs = {session.get_inputs()[0].name: datain}
outputs = session.run(None, inputs)
check_matrix_equal(outputs[0], dataout, 1e-4, dataroot, "onnx")

###### 检查预处理函数preprocess
datainsrc = preprocess(image, 256, 256)
check_matrix_equal(datainsrc, datain, 1e-4, dataroot, "preprocess")

###### 检查后处理函数preprocess
predsrc = postprocess(dataout)
check_matrix_equal(predsrc, pred, 1e-4, dataroot, "postprocess")


