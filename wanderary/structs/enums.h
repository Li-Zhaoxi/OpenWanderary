#pragma once
#include <set>

#include <wanderary/utils/enum_traits.h>

ENUM_NUMBERED_REGISTER(SensorNameID,                                   //
                       ((kUnknown, 0, "unknown"))                      //
                       ((kCameraUnknown, 1, "camera_unknown"))         //
                       ((kCameraFront, 2, "camera_front"))             //
                       ((kCameraFrontLeft, 3, "camera_front_left"))    //
                       ((kCameraFrontRight, 4, "camera_front_right"))  //
                       ((kCameraSideLeft, 5, "camera_side_left"))      //
                       ((kCameraSideRight, 6, "camera_side_right"))    //
                       ((kCameraRearLeft, 7, "camera_rear_left"))      //
                       ((kCameraRear, 8, "camera_rear"))               //
                       ((kCameraRearRight, 9, "camera_rear_right"))    //
                       ((kLaserUnknown, 10, "laser_unknown"))          //
                       ((kLaserTop, 11, "laser_top"))                  //
                       ((kLaserFront, 12, "laser_front"))              //
                       ((kLaserSideLeft, 13, "laser_side_left"))       //
                       ((kLaserSideRight, 14, "laser_side_right"))     //
                       ((kLaserRear, 15, "laser_rear"))                //
)
ENUM_CONVERSION_REGISTER(SensorNameID, SensorNameID::kUnknown, "unknown")

namespace wdr {

class SensorUtils {
 public:
  static bool is_camera(SensorNameID id) {
    return camera_ids_.find(id) != camera_ids_.end();
  }
  static bool is_laser(SensorNameID id) {
    return laser_ids_.find(id) != laser_ids_.end();
  }

 private:
  inline static const std::set<SensorNameID> camera_ids_ = {
      SensorNameID::kCameraUnknown,   SensorNameID::kCameraFront,
      SensorNameID::kCameraFrontLeft, SensorNameID::kCameraFrontRight,
      SensorNameID::kCameraSideLeft,  SensorNameID::kCameraSideRight,
      SensorNameID::kCameraRearLeft,  SensorNameID::kCameraRear,
      SensorNameID::kCameraRearRight,
  };
  inline static const std::set<SensorNameID> laser_ids_ = {
      SensorNameID::kLaserUnknown,   SensorNameID::kLaserTop,
      SensorNameID::kLaserFront,     SensorNameID::kLaserSideLeft,
      SensorNameID::kLaserSideRight, SensorNameID::kLaserRear,
  };
};

}  // namespace wdr
