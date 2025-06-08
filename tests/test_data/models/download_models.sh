
model_dir=$(cd $(dirname $0) && pwd)
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/AAA_RDK_YOLO/yolov8_detect_nv12/yolov8n_detect_bayese_640x640_nv12_modified.bin -P $model_dir
