import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from scipy.special import softmax


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


imgpath = "examples/modules/CONIC_161.png"
modelpath = "examples/modules/unet.bin"
npyinpath = "bpumatin.npy"
npyoutpath = "bpumatout.npy"

models = dnn.load(modelpath)
model_h, model_w = get_hw(models[0].inputs[0].properties)
print(model_h, model_w)
print(models[0].inputs[0].properties.tensor_type)
print(dir(models[0].inputs[0].properties))

img = cv2.imread(imgpath)
img = cv2.resize(img, (model_w, model_h))

imgincpp = np.load(npyinpath)
imgoutcpp = np.load(npyoutpath)

print(img.shape, img.dtype)
print(imgincpp.shape, imgincpp.dtype)

imgdiff = np.abs(img.astype(np.float32) - imgincpp.astype(np.float32))
print(np.max(imgdiff), np.min(imgdiff))




t1 = cv2.getTickCount()
outputs = models[0].forward(img)
t2 = cv2.getTickCount()
print('time consumption {0} ms'.format((t2-t1)*1000/cv2.getTickFrequency()))


outputs = outputs[0].buffer
print(len(outputs), outputs.shape)

print(outputs.shape, outputs.dtype)
print(imgoutcpp.shape, imgoutcpp.dtype)
imgdiff = np.abs(outputs.astype(np.float32) - imgoutcpp.astype(np.float32))
print(np.max(imgdiff), np.min(imgdiff))

print(outputs[0, 1, :, :])

y_list = softmax(outputs, axis = 1)[0, 1, :, :]
print(y_list)
y_list = (y_list > 0.5).astype(np.uint8) * 255


cv2.imwrite('pred_py.png', y_list)