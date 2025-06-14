import time

import common  # noqa
import cv2
import numpy as np
import pytest

from wanderary import (AutoScopeTimer, GlobalTimerManager, TimerManager,
                       cvtBGR2NV12, cvtNV12ToYUV444)


def test_convertor_bgr2nv12():
    imgpath = "../test_data/media/zidane.jpg"
    gtpath = "../test_data/utils/zidane_nv12.png"

    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imggt = cv2.imread(gtpath, cv2.IMREAD_COLOR)

    nv12 = cvtBGR2NV12(img)
    yuv444 = cvtNV12ToYUV444(nv12, img.shape[1], img.shape[0])
    res = cv2.cvtColor(yuv444, cv2.COLOR_YUV2BGR)

    np.testing.assert_array_equal(res, imggt)


def test_timemanager_record():
    tm = TimerManager()
    tm.start("stage1")
    time.sleep(0.1)
    tm.start("stage2")
    time.sleep(0.2)
    tm.stop("stage2")
    tm.stop("stage1")

    dt1 = tm.getDuration("stage1")
    dt2 = tm.getDuration("stage2")

    tm.printStatistics()

    np.testing.assert_almost_equal(dt1, 300, decimal=1)
    np.testing.assert_almost_equal(dt2, 200, decimal=1)


def test_timemanager_autoscope():
    tm = TimerManager()
    with AutoScopeTimer("stage1", tm):
        time.sleep(0.1)
        with AutoScopeTimer("stage2", tm):
            time.sleep(0.2)

    dt1 = tm.getDuration("stage1")
    dt2 = tm.getDuration("stage2")
    tm.printStatistics()

    np.testing.assert_almost_equal(dt1, 300, decimal=-1)
    np.testing.assert_almost_equal(dt2, 200, decimal=-1)


def global_test_fun1():
    with AutoScopeTimer("stage1", GlobalTimerManager()):
        time.sleep(0.1)


def global_test_fun2():
    with AutoScopeTimer("stage2", GlobalTimerManager()):
        time.sleep(0.2)


def test_timemanager_globaltimer():
    global_test_fun1()
    global_test_fun2()

    tm = GlobalTimerManager()
    tm.printStatistics()

    dt1 = tm.getDuration("stage1")
    dt2 = tm.getDuration("stage2")

    np.testing.assert_almost_equal(dt1, 100, decimal=-1)
    np.testing.assert_almost_equal(dt2, 200, decimal=-1)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
