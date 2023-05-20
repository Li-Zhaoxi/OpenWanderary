import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from networks.unet import UNet
import onnx

# 1. 加载Pytorch模型
dataroot = "data/unet"
modelpath = os.path.join(dataroot, "checkpoint_0.pth.tar")
net = UNet(3, 2) # 定义模型，参数1表示输入图像是3通道，参数2表示类别个数
net = torch.nn.DataParallel(net).cpu() # ※这行代码不能删，否则模型参数无法加载成功
state_dict = torch.load(modelpath)
net.load_state_dict(state_dict["state_dict"]) # 把参数拷贝到模型中
net.eval()  # ※这个要有

# 2. 转换ONNX
onnxpath = os.path.join(dataroot, "unet.onnx") # 定义onnx文件保存目录
im = torch.randn(1, 3, 256, 256).cpu() # 定义输入变量，维度重要，内容不重要
# 下面是转onnx的基本配置，为了能够用在BPU上，按照下面这个方式配置参数即可
# 对于多输入输出的模型，参考连接：https://www.cnblogs.com/WenJXUST/p/16334151.html
# 下面这个Warning可以忽略不管
# Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
torch.onnx.export(net.module,
                  im,
                  onnxpath,
                  verbose=False,
                  training=torch.onnx.TrainingMode.EVAL,
                  do_constant_folding=True,
                  input_names=['images'],
                  output_names=['output'],
                  dynamic_axes=None,
                  opset_version=11)

# 3. 检查ONNX，如果ONNX有问题，这里会输出一些日志
print('Start check onnx')
model_onnx = onnx.load(onnxpath)
onnx.checker.check_model(model_onnx)