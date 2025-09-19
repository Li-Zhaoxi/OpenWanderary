import json

import common  # noqa
import cv2
import pytest

from wanderary import (Box2D, BPUNets, CheckUnorderedBoxes, ProcessManager,
                       ProcessRecorder)


def test_dnn_yolov8():
    imgpath = "../test_data/media/zidane.jpg"
    binpath = "../test_data/models/yolov8n_detect_bayese_640x640_nv12_modified.bin" # noqa
    model_name = "yolov8n_detect_bayese_640x640_nv12"
    precfg = {
      "manager_name": "pre-process",
      "processes": [
        {
          "name": "FormatImage",
          "config": {
            "type": "letter_box",
            "width": 640,
            "height": 640,
            "cvt_nv12": True
          }
        }
      ]
    }
    postcfg = {
      "manager_name": "yolo-post-process",
      "processes": [
        {
          "name": "ConvertYoloFeature",
          "config": {
            "class_num": 80,
            "reg_num": 16,
            "nms_thres": 0.7,
            "score_thres": 0.25,
            "box_scales": {"1": 8, "3": 16, "5": 32}
          }
        }
      ]
    }

    recorder = ProcessRecorder()
    preproc = ProcessManager(precfg)
    postproc = ProcessManager(postcfg)
    nets = BPUNets(binpath)
    recorder.dequant_scales = nets.GetDequantScales(model_name)  # 记录反量化信息
    img = cv2.imread(imgpath)  # shape: 720x1280x3

    # 输出矩阵，并记录变换信息
    prepare_data = preproc.Forward(img, recorder)  # shape: 1x614400
    # 输出特征List
    net_feats = nets.forward(model_name, [prepare_data])  # len: 6
    box2ds = postproc.Forward2D(net_feats, recorder)
    print(box2ds, len(box2ds))

    # 加载真值验证
    gtpath = "../test_data/process/yolov8_gt_box2ds.json"
    gt_box2ds = []
    with open(gtpath, "r") as f:
        gt_data = json.load(f)
        for boxdata in gt_data:
            gt_box2ds.append(Box2D.load(boxdata))
    assert CheckUnorderedBoxes(box2ds, gt_box2ds, 1e-4), "Test Failed!"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
