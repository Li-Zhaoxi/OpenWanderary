from hobot_vio import libsrcampy as srcampy
import cv2
import numpy as np

class ImageShow(object):
    # 初始化，screen_w和screen_h表示HDMI输出支持的显示器分辨率
    def __init__(self, screen_w = 1920, screen_h = 1080):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.disp = srcampy.Display()
        self.disp.display(0, screen_w, screen_h)
     
    # 结束显示
    def close(self):
        self.disp.close()
        
    # 显示图像，输入image即可
    def show(self, image, wait_time=0):
        imgShow = self.putImage(image, self.screen_w, self.screen_h)
        imgShow_nv12 = self.bgr2nv12_opencv(imgShow)
        self.disp.set_img(imgShow_nv12.tobytes())
    
    # 私有函数，将图像数据转换为用于HDMI输出的数据
    @classmethod
    def bgr2nv12_opencv(cls, image):
        height, width = image.shape[0], image.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
    
    # 图像数据在显示器最大化居中
    @classmethod
    def putImage(cls, img, screen_width, screen_height):
        if len(img.shape) == 2:
            imgT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            imgT = img
        irows, icols = imgT.shape[0:2]
        scale_w = screen_width * 1.0/ icols
        scale_h = screen_height * 1.0/ irows
        final_scale = min([scale_h, scale_w])
        final_rows = int(irows * final_scale)
        final_cols = int(icols * final_scale)
        print(final_rows, final_cols)
        imgT = cv2.resize(imgT, (final_cols, final_rows))
        diff_rows = screen_height - final_rows
        diff_cols = screen_width - final_cols
        img_show = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        img_show[diff_rows//2:(diff_rows//2+final_rows), diff_cols//2:(diff_cols//2+final_cols), :] = imgT
        return img_show
