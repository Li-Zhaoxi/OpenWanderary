import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import cv2
import numpy as np
import torch
from networks.unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_torch(img, modelh, modelw):
  transform = [A.Resize(modelh, modelw),
              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
              ToTensorV2()]
  





modelpath = "D:/01 - datasets/008 - 2DMRACerebrovascular/Experiments/baseline/unet/fold_0/checkpoints/checkpoint_0.pth.tar"

net = UNet(3, 2, 2)
net = torch.nn.DataParallel(net).cpu()
state_dict = torch.load(modelpath)
net.load_state_dict(state_dict["state_dict"])
net.eval()