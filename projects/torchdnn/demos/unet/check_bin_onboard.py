### 注意：该代码仅能在开发板中运行
import numpy as np
import os
import onnxruntime
from prepare_functions import check_matrix_equal
from hobot_dnn import pyeasy_dnn as dnn

dataroot = "data/unet"
data = np.load(os.path.join(dataroot, "unet_checkstage2.npz"))

datain = data["datain"]
dataout = data["dataout"]

###### 检查量化BIN有效性
binpath = os.path.join(dataroot, "model_output/unet.bin")
models = dnn.load(binpath)

outputs = models[0].forward(datain)

check_matrix_equal(outputs[0].buffer, dataout, 1e-4, dataroot, "onnxonboard")