### 注意：该代码仅能在OE docker中运行
import numpy as np
import cv2
import os
from prepare_functions import get_bgr_image, preprocess_onboard, postprocess
# horizon_nn 是在docker中才有的包
from horizon_nn import horizon_onnxruntime
from horizon_nn import horizon_onnx

dataroot = "/data/horizon_x3/data"
imgpath = os.path.join(dataroot, "mra_img_12.jpg")
onnxpath = os.path.join(dataroot, "model_output", "unet_quantized_model.onnx")

# 加载图像和ONNX模型
img = get_bgr_image(imgpath)
session = horizon_onnxruntime.InferenceSession(onnxpath)

# 板端预处理函数
datain = preprocess_onboard(img, 256, 256) # 1x256x256x3
# 构建输入并推理，记得要转为int8
inputs = {session.get_inputs()[0].name: (datain.astype(np.int32) - 128).astype(np.int8)}
outputs = session.run(None, inputs)

# 后处理并保存结果
pred = postprocess(outputs[0])
cv2.imwrite(os.path.join(dataroot, "pred_quantized_onnx.png"), pred[0])

# 保存校验数据
data = {"image": img,
        "datain": datain,
        "dataout": outputs[0],
        "pred": pred}
np.savez(os.path.join(dataroot, "unet_checkstage2.npz"), **data)

