
开源项目OpenWanderary[wɒndərəri]，拥抱多传感器，用于提高嵌入式系统部署效率，所有功能来源于项目，应用于项目。该项目重点在于三点：
- **Open** Source。开源节流，所用即所得。用了啥就会记录什么。
- Integrate **Wander**ing Sensors/Device。集成多设备多传感器所用的相关库，以及算法。
- Libr**ary**。该项目重点以库为主，examples和projects给出了该库的各种解决方案。


设计目的，在机器人开发中会有很多重复的工作，这些工作造一次轮子就足够了，无需重复构造。下面有一些设计规范：

- 尽可能不构造新的数据类型，降低学习成本。所用数据依赖常规库Eigen, OpenCV, PCL。
- wdr库内部不会将其他人的库直接拷贝粘贴进核心内容中，所依赖的其他仓库将会以第三方库的形式的进行封装。
- 不依赖硬件。当前库以Linux为主，对于依赖特定硬件的功能，可通过配置CMAKE文件进行配置。
- 完全开源，但仅有一个要求：应用wdr库时请以动态链接方式连接，有优化项需求希望以pull requrest形式更新，欢迎各位贡献代码。
- 代码设计风格参考Effective C++形式。这部分也是一边学习一边优化。
- 编译时候以动态链接库so文件为主

项目规划：
1. 工具正式对外发布。条件：WDR代码量>1w行，可支持已有的三个项目。
2. 联合其他人贡献代码。代码量超过0.5w行，项目支持数 or 函数增加数超过阈值，则可发布下一版本。需要支持更多传感器/更多设备比如无人机等等。

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```


CircularProc：圆结构处理相关，包含检测，测量，合作靶标检测相关，以Ptr模式存储

需要咨询cmake构建方式。

LOG以glog为主


sudo apt-get install libboost-filesystem1.71-dev