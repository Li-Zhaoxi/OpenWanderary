import json

import common  # noqa
import cv2
import pytest

from wanderary import (Box2D, Box2DDrawer, CheckUnorderedBoxes, ImageCropROIs,
                       StatisticsTimeManager, TimerManager, YOLOv8)


def test_yolov8_largeimage():
    imgpath = "../test_data/utils/street.jpg"
    cfgpath = "../test_data/apps/yolov8_config.json"
    gtpath = "../test_data/utils/street_box2d_gt.json"
    names_path = "../test_data/tiny_coco/type_names.json"
    binpath = "../test_data/models/yolov8x_detect_bayese_640x640_nv12_modified.bin" # noqa

    # 构造GT
    with open(gtpath, "r") as f:
        gtdata = json.load(f)
    box2ds_gt = [Box2D.load(item) for item in gtdata]

    # 初始化
    class_names = [""] * 80
    with open(names_path, "r") as f:
        names = json.load(f)
        for i, name in enumerate(names):
            class_names[int(i)] = name
    drawer = Box2DDrawer(80, class_names)
    with open(cfgpath, "r") as f:
        cfg = json.load(f)
        cfg["model_path"] = binpath
    yolo = YOLOv8("yolov8", cfg, 1)
    img = cv2.imread(imgpath)

    stats_manager = StatisticsTimeManager()
    for idx in range(10):
        rois = ImageCropROIs(img.shape[1::-1], (640, 640), (640, 640), False)
        time_manager = TimerManager()
        time_manager.start("full-pipeline")
        bbox2ds = yolo.run(img, rois, stats_manager)
        time_manager.stop("full-pipeline")
        stats_manager.add(time_manager)

        assert CheckUnorderedBoxes(bbox2ds, box2ds_gt, 1e-4), "Test Failed!"

        if idx == 0:
            vis = drawer.draw(bbox2ds, img)
            cv2.imwrite("pyvis_yolov8_large_image.jpg", vis)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
