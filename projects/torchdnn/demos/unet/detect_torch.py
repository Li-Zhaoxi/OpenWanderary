import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
import torch
from networks.unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2



# 基于torch的数据预处理：输入维度[h,w,c]，返回的数据排布为[c,h,w]
def preprocess_torch(img, modelh, modelw):
  transform = A.Compose([A.Resize(modelh, modelw),
              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
              ToTensorV2()])
  return transform(image = img)

# 基于torch的数据后处理: 输入[b,c,h,w]，输出[b,2,h,w]
def postprocess_torch(dataout: torch.Tensor) -> np.ndarray:
  y = torch.nn.Softmax(dim=1)(dataout)[:, 1].cpu().detach().numpy()
  pred = (y > 0.5).astype(np.uint8) * 255
  return pred


dataroot = "data/unet"

# 1. Pytorch 模型
modelpath = os.path.join(dataroot, "checkpoint_0.pth.tar")

# 2. 模型加载
net = UNet(3, 2, 2)
net = torch.nn.DataParallel(net)
state_dict = torch.load(modelpath)
net.load_state_dict(state_dict["state_dict"])
net.eval()
net = net.module.cpu() # 一定要指定为CPU 

# 3. 图像数据(模型以RGB为输入)
imgpath = os.path.join(dataroot, "mra_img_12.jpg")
img = cv2.imread(imgpath)
if len(img.shape) == 2:
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. 数据预处理
datain = preprocess_torch(img, 256, 256)['image'] # cxhxw
datain = torch.unsqueeze(datain, dim=0).cpu() # 1xcxhxw

# 5. 模型推理：推理输出维度[1,2,256,256]
dataout = net(datain)

# 6. 数据后处理：pred维度[1,256,256]
pred = postprocess_torch(dataout)
print(pred.shape, type(pred))
for j in range(pred.shape[0]):
  cv2.imwrite(os.path.join(dataroot, f"pred_batch_{j}.png"), pred[j].astype(np.uint8))

# 7. 保存校验数据
data = {"image": img,
        "datain": datain.cpu().detach().numpy(),
        "dataout": dataout.cpu().detach().numpy(),
        "pred": pred}
np.savez(os.path.join(dataroot, "unet_checkstage1.npz"), **data)