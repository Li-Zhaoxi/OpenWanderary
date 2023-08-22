#include <iostream>
#include <Eigen/Eigen>

void init_mats(std::vector<Eigen::Matrix4f> &mats);
void eigen_inverse();

int main(int argc, char **argv)
{
  eigen_inverse();
  return 0;
}

void init_mats(std::vector<Eigen::Matrix4f> &mats)
{
  mats.resize(4);

  mats[0] << 0.999981, -0.00415212, -0.00466134, 0.71,
      0.00414895, 0.999991, -0.000689198, 0,
      0.00466416, 0.000669844, 0.999989, 1.85264,
      0, 0, 0, 1;

  mats[1] << -0.999819, -0.0184469, -0.00462907, 0.800784,
      -0.00444588, -0.00995929, 0.999941, 0.601399,
      -0.0184919, 0.99978, 0.00987548, 1.63478,
      0, 0, 0, 1;

  mats[2] << 0.00987783, 0.00454281, -0.999941, -0.910941,
      -0.999715, -0.0216873, -0.00997413, -0.002741,
      -0.0217313, 0.999754, 0.00432729, 1.19948,
      0, 0, 0, 1;

  mats[3] << 0.999939, 0.00712378, 0.00849454, 0.811098,
      0.00856168, -0.00944788, -0.999919, -0.620238,
      -0.00704294, 0.99993, -0.00950829, 1.67974,
      0, 0, 0, 1;
}

void eigen_inverse()
{
  std::vector<Eigen::Matrix4f> mats;
  init_mats(mats);

  const int num = mats.size();
  std::vector<Eigen::Matrix4f> T_sensors2base(num);
  auto T_trans = T_sensors2base[0].inverse(); // Wrong
  // Eigen::Matrix4f T_trans = T_sensors2lidartop[0].inverse(); // Right
  for (int k = 0; k < num; k++)
  {
    std::cout << k << " src: " << T_sensors2base[k] << std::endl;
    std::cout << "T_trans: " << T_trans << std::endl;
    T_sensors2base[k] = T_trans * T_sensors2base[k];
    std::cout << k << " dst: " << T_sensors2base[k] << std::endl;
  }
}