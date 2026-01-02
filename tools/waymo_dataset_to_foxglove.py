import argparse
import logging
import sys

# isort: off
from wanderary import MCAPWriter, MultiModalFrame, SensorNameID2str
# isort: on

import tensorflow as tf
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def parse_args(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--save-raw", action="store_true")

    return parser.parse_args(argv[1:])


def main(*argv):
    if not argv:
        argv = list(sys.argv)
    args = parse_args(*argv)

    record_path = args.record_path
    save_path = args.save_path
    save_raw = args.save_raw

    dataset = tf.data.TFRecordDataset(record_path)
    mcap_writer = MCAPWriter(save_path)

    # 定义topic
    topic_name_raw = "waymo_dataset/raw"
    topic_name_image = "waymo_dataset/image_"
    topic_name_box2d = "waymo_dataset/box2d_"

    # 遍历每一帧数据
    seq_count = 0
    for data in tqdm(dataset):
        mmframe = MultiModalFrame()

        # 保存raw数据
        mcap_writer.WriteWaymoFrame(
          topic_name_raw if save_raw else "",
          bytearray(data.numpy()), seq_count, mmframe)

        # 保存图像
        for chl, frame in mmframe.camera_frames().items():
            sensor_name = SensorNameID2str(chl)
            mcap_writer.WriteImage(
                topic_name_image + sensor_name, frame, seq_count)
            mcap_writer.WriteImageBox2Ds(
                topic_name_box2d + sensor_name, frame, seq_count)

        seq_count += 1
        break

    mcap_writer.close()


if __name__ == "__main__":
    main(*sys.argv)
