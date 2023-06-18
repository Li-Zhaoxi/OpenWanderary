### 注意：该代码仅能在开发板中运行
import numpy as np
import cv2
import os
from prepare_functions import get_bgr_image, preprocess_onboard, postprocess
from hobot_dnn import pyeasy_dnn as dnn


# 记得要安装一些包
# sudo pip3 install scipy opencv-contrib-python

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


dataroot = "data/unet"
imgpath = os.path.join(dataroot, "mra_img_12.jpg")
binpath = os.path.join(dataroot, "model_output/unet.bin")


# 加载图像和BIN模型
img = get_bgr_image(imgpath)
models = dnn.load(binpath)

# 图像数据预处理，这里对几个地方进行解释：
# models[0]：BPU支持加载多个模型，这里只有一个unet，因此[0]表示访问第一个模型
# inputs[0]：模型输入可能多种，unet的输入只有一个，因此[0]表示获取第一个输入的参数
model_h, model_w = get_hw(models[0].inputs[0].properties)



datain = preprocess_onboard(img, model_h, model_w) # 1x256x256x3


# 模型推理：相比于onnx推理，这里不用再重新构造一个inputs
t1 = cv2.getTickCount()
outputs = models[0].forward(datain)
t2 = cv2.getTickCount()
print('time consumption {0} ms'.format((t2-t1)*1000/cv2.getTickFrequency()))


pro = models[0]
print(type(pro.name), pro.name)
print(type(pro.estimate_latency), pro.estimate_latency)

# 后处理并保存结果，这里对几个地方进行解释
# outputs数据类型为tuple，每个元素的数据类型是pyDNNTensor
# 因此想要获取输出的矩阵的话，需要调用buffer
pred = postprocess(outputs[0].buffer) # outputs[0].buffer: (1, 2, 256, 256)
cv2.imwrite(os.path.join(dataroot, "pred_bin.png"), pred[0])