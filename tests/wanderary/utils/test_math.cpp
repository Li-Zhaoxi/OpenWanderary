#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

std::vector<Eigen::Matrix4d> getTransformations() {
  std::vector<Eigen::Matrix4d> mats(4);
  mats[0] << 0.999981, -0.00415212, -0.00466134, 0.71,  //
      0.00414895, 0.999991, -0.000689198, 0,            //
      0.00466416, 0.000669844, 0.999989, 1.85264,       //
      0, 0, 0, 1;

  mats[1] << -0.999819, -0.0184469, -0.00462907, 0.800784,  //
      -0.00444588, -0.00995929, 0.999941, 0.601399,         //
      -0.0184919, 0.99978, 0.00987548, 1.63478,             //
      0, 0, 0, 1;

  mats[2] << 0.00987783, 0.00454281, -0.999941, -0.910941,  //
      -0.999715, -0.0216873, -0.00997413, -0.002741,        //
      -0.0217313, 0.999754, 0.00432729,
      1.19948,  //
      0, 0, 0, 1;

  mats[3] << 0.999939, 0.00712378, 0.00849454, 0.811098,  //
      0.00856168, -0.00944788, -0.999919, -0.620238,      //
      -0.00704294, 0.99993, -0.00950829,
      1.67974,  //
      0, 0, 0, 1;

  return mats;
}

/**
 * \brief 介绍Eigen库中的推断陷阱
 * \note 参考博客: https://blog.csdn.net/Zhaoxi_Li/article/details/128873970
 */
TEST(Eigen, AutoInverse) {
  const auto raw_mats = getTransformations();
  {  // 用auto 来存放逆矩阵
    std::vector<Eigen::Matrix4d> T_sensors2base = raw_mats;
    const auto T_trans = T_sensors2base[0].inverse();
    const int num = T_sensors2base.size();
    for (int i = 0; i < num; i++) {
      auto &T = T_sensors2base[i];
      LOG(INFO) << "idx: " << i << ", src mat: \n"
                << T << "\ninv mat: \n"
                << T_trans;
      T = T_trans * T;
      LOG(INFO) << "idx: " << i << ", dst mat: \n" << T;
    }

    EXPECT_TRUE(T_sensors2base[0].isApprox(Eigen::Matrix4d::Identity(), 1e-5));
    EXPECT_TRUE(T_sensors2base[1].isApprox(raw_mats[1], 1e-5));
    EXPECT_TRUE(T_sensors2base[2].isApprox(raw_mats[2], 1e-5));
    EXPECT_TRUE(T_sensors2base[3].isApprox(raw_mats[3], 1e-5));
  }

  {  // 用具体类型来存放逆矩阵
    std::vector<Eigen::Matrix4d> T_sensors2base = raw_mats;
    const Eigen::Matrix4d T_trans = T_sensors2base[0].inverse();
    const int num = T_sensors2base.size();
    for (int i = 0; i < num; i++) {
      auto &T = T_sensors2base[i];
      LOG(INFO) << "idx: " << i << ", src mat: \n"
                << T << "\ninv mat: \n"
                << T_trans;
      T = T_trans * T;
      LOG(INFO) << "idx: " << i << ", dst mat: \n" << T;
    }

    EXPECT_TRUE(T_sensors2base[0].isApprox(Eigen::Matrix4d::Identity(), 1e-5));
    EXPECT_FALSE(T_sensors2base[1].isApprox(raw_mats[1], 1e-5));
    EXPECT_FALSE(T_sensors2base[2].isApprox(raw_mats[2], 1e-5));
    EXPECT_FALSE(T_sensors2base[3].isApprox(raw_mats[3], 1e-5));
  }
}
