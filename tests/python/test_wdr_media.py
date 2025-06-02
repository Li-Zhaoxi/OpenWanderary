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


@pytest.mark.skip(reason="如果有支持的USB摄像头, 可以取消skip")
def test_usb_video():
    """
    利用USB摄像头, 读取MJPG视频流, 并使用软硬解码获取图像
    """
    imgh, imgw = 2448, 3264
    flag_hard_decode = True

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgh)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if flag_hard_decode:
        codec = MediaCodecJpg(
          MediaCodecID.kMJPEG, False, imgw, imgh)
        codec.init()

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        data = cap.grab()
        if not data:
            continue
        ret, enc_data = cap.retrieve()
        if not ret:
            continue

        t1 = cv2.getTickCount()
        if flag_hard_decode:
            yuv = codec.process(enc_data)
            if yuv is None:
                continue
            dec_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        else:
            dec_img = cv2.imdecode(enc_data, cv2.IMREAD_COLOR)
        t2 = cv2.getTickCount()
        consume_time = (t2 - t1) / cv2.getTickFrequency()
        print(
          f"decode time: {consume_time:.3f} ms, resolution: {dec_img.shape}")

        small_img = cv2.resize(dec_img, (imgw // 4, imgh // 4))
        cv2.imshow("frame", small_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if flag_hard_decode:
        codec.close()

    cap.release()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
