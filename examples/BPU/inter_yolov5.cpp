
#include "YOLO/yolov5.h"


int main(int argc, char **argv)
{
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Begin test_interface_yolov5_bpu" << std::endl;
    // std::string modelbinpath = "config/yolov5s_cat.bin";
    std::string modelbinpath = "config/yolov5s.bin";
    std::string imagepath = "examples/data/20220902222444.jpg";
    
    std::vector<cv::Mat> images(1);
    images[0] = cv::imread(imagepath);

    std::cout << "start init YoloV5" << std::endl;
    YoloV5 yolo(modelbinpath);
    std::cout << "start initModel" << std::endl;
    CV_Assert(yolo.initModel(modelbinpath) == 0);

    std::cout << "start prepareImage" << std::endl;
    CV_Assert(yolo.prepareImage(images) == 0);

    std::cout << "start interface" << std::endl;
    CV_Assert(yolo.interface() == 0);

    return 0;
}