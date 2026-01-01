#pragma once

#include <wanderary/structs/frame.h>
#include <waymo_open_dataset/dataset.pb.h>

namespace wdr::msg {

// 填充Frame的Image相关属性
void ConvertCameraImageMsgToFrame(const waymo::open_dataset::CameraImage &frame,
                                  ImageFrame *frame2d);

// 将Waymo的Box构建为Box2D
void ConvertLabelBoxMsgToBox2D(const waymo::open_dataset::Label::Box &msg,
                               Box2D *box2d);

// 补充标记的2D Box
void ConvertCameraLabelsMsgToFrame(
    const waymo::open_dataset::CameraLabels &frame, ImageFrame *frame2d);

// 将Waymo的Frame转换为多模帧
void ConvertWaymoFrameMsgToMMFrame(const waymo::open_dataset::Frame &frame,
                                   MultiModalFrame *mmframe);

}  // namespace wdr::msg
