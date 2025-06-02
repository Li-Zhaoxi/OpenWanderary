import common  # noqa
import cv2
import numpy as np
import pytest

from wanderary import MediaCodecID, MediaCodecJpg, cvtBGR2NV12


def test_jpg_encode():
    imgpath = "../test_data/media/zidane.jpg"
    gtpath = "../test_data/media/zidane_encode.bin"

    gt_enc = np.fromfile(gtpath, dtype=np.uint8)

    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    nv12 = cvtBGR2NV12(img)
    codec = MediaCodecJpg(
      MediaCodecID.kMJPEG, True, img.shape[1], img.shape[0])
    codec.init()
    res = codec.process(nv12)
    print(res.shape)
    codec.close()

    assert res is not None
    np.testing.assert_array_equal(res[0], gt_enc)


def test_jpg_decode():
    imgpath = "../test_data/media/zidane.jpg"
    gtpath = "../test_data/media/zidane_decode.png"
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    gtimg = cv2.imread(gtpath, cv2.IMREAD_COLOR)

    enc_buf = cv2.imencode(".jpg", img)[1]
    print(enc_buf.shape)
    codec = MediaCodecJpg(
      MediaCodecID.kMJPEG, False, img.shape[1], img.shape[0])
    codec.init()
    res = codec.process(enc_buf)
    print(res.shape)
    codec.close()

    dec_img = cv2.cvtColor(res, cv2.COLOR_YUV2BGR_NV12)

    np.testing.assert_array_equal(dec_img, gtimg)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
