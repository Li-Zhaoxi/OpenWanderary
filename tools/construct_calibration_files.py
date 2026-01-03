import argparse
import logging
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def parse_args(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--num", type=int, default=50)

    return parser.parse_args(argv[1:])


def main(*argv):
    if not argv:
        argv = list(sys.argv)
    args = parse_args(*argv)

    data_path = args.data_path
    save_dir = args.save_dir
    model_size = args.model_size.split(",")
    modelw, modelh = int(model_size[0]), int(model_size[1])

    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(data_path):
        # 遍历目录下的所有文件, 如果文件名以".jpg"结尾, 则将其复制到目标目录
        filenames = []
        for filename in os.listdir(data_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filenames.append(filename)
        step = len(filenames) // args.num + 1
        for i in tqdm(range(0, len(filenames), step)):
            filename = filenames[i]
            img = cv2.imread(os.path.join(data_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR->RGB
            img = cv2.resize(img, (modelw, modelh))
            img = img.transpose(2, 0, 1)  # HWC->CHW
            img = np.expand_dims(img, 0)  # 1CHW
            # 校验数据预处理函数, 注意输入的img是RGB通道
            savepath = os.path.join(
                save_dir, filename.split(".")[0] + ".rgbchw")
            img.astype(np.float32).tofile(savepath)
    else:
        raise Exception(f"Unsupported data type: {data_path}")


if __name__ == "__main__":
    main(*sys.argv)
