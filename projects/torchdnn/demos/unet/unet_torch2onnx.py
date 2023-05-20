import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))


import cv2
import numpy as np
import torch
from networks.unet import UNet
import onnx

modelpath = "D:/01 - datasets/008 - 2DMRACerebrovascular/Experiments/baseline/unet/fold_0/checkpoints/checkpoint_0.pth.tar"

net = UNet(3, 2, 2)
net = torch.nn.DataParallel(net).cpu()
state_dict = torch.load(modelpath)
net.load_state_dict(state_dict["state_dict"])
net.eval()


onnxpath = "unet.onnx"
im = torch.randn(1, 3, 256, 256).cpu()
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

# Checks
print('Start check onnx')
model_onnx = onnx.load(onnxpath)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model