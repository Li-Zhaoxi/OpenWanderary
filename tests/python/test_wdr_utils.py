import common  # noqa
import cv2
import numpy as np

from wanderary import cvtBGR2NV12, cvtNV12ToYUV444


def test_convertor_bgr2nv12():
    imgpath = "../test_data/media/zidane.jpg"
    gtpath = "../test_data/utils/zidane_nv12.png"

    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imggt = cv2.imread(gtpath, cv2.IMREAD_COLOR)

    nv12 = cvtBGR2NV12(img)
    yuv444 = cvtNV12ToYUV444(nv12, img.shape[1], img.shape[0])
    res = cv2.cvtColor(yuv444, cv2.COLOR_YUV2BGR)

    np.testing.assert_array_equal(res, imggt)


if __name__ == "__main__":
    test_convertor_bgr2nv12()
    # pytest.main(["-s", __file__])
