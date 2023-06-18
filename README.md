 
开源项目OpenWanderary[wɒndərəri]，拥抱多传感器，核心在于提高基于Linux的嵌入式系统的研发&部署效率。所有功能来源于项目，应用于项目。

自己做一些嵌入式开发时候，总会遇到一些问题，每家的硬件/设备都有自己的一套API，在学习测试这些API上都要花费较多时间，而且项目开发中总会有一些基础功能都要重新造一遍，这些功能并不难，但开发+测试总会占用较多的时间。因此我也经常在想，如果有一个好点的基础库，是不是可以省下更多的时间来研发更多有趣的应用。

因此，我打算一边学习，一遍将学到的基本功能记录下来，过去7年，攒了一堆零散的功能，也都会集成在这里，希望能用几年的时间，打造一个大家都喜欢的库。项目OpenWanderary(缩写WDR)重点为以下几点，后面新增的功能都是围绕这几点展开：
- **Open** Source。WDR中扩展的库都是完全开源的，提供足够的灵活性。
- Integrate **Wander**ing Sensors/Device。集成多设备/多传感器的相关库，以及使用这些设备的相关算法。
- Libr**ary**。WDR生成的是动态库，拿过来就直接使用。Examples和Projects给出了该库的各种示例/解决方案。

轮子造一次就足够了，无需重复构造。下面有一些设计规范：
- **尽可能不构造新的数据类型**，降低学习成本。所用数据依赖常规库Eigen, OpenCV, PCL。新的数据类型就算构建，也是以降低使用成本为目的。
- **不依赖硬件**。当前库以Linux为主，如果有些硬件并不需要，可通过配置CMAKE文件来决定是否编译。
- **完全开源**。希望各位使用WDR时以动态链接方式使用，不要赋值其中的代码到自己的项目中。有优化项需求希望以pull requrest形式更新，欢迎各位贡献代码。
- **提供优质的开发API**。整体代码，以C++17为基础进行开发，开发过程中参考了《Effective C++》中的一些条款。这部分也是一边学习一边优化，优化代码规范，架构规范。


# 代码编译

依赖库安装，可能也有一些没在这里体现，比如glog,gflags等
```
sudo apt-get install libopencv-dev libboost-filesystem1.71-dev
```
编译代码，内存多可以试试make -j6。交叉编译还没整明白ㄒoㄒ
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

我这里编译代码用的是vscode的cmake插件。

# API文档

我还没来得及写ε=ε=ε=┏(゜ロ゜;)┛

文档注释规范啥的完全没学过，得后面慢慢完善这部分的功能了

目前相关功能的使用，暂时先通过博客"[俺的CSDN博客](https://blog.csdn.net/Zhaoxi_Li?type=blog)"提供的一些示例学习下吧。


# 版本管理

期望每个月都能保证一个版本的更新，今年奔着WDR代码增加1w行代码为目标，先试试水（从历史经验看，1w+以上的代码已具有一定的稳定性）。

以版本号：v1.v2.v3-v4为例
- **v1为主版本号**。增加/修改了重要的功能架构，若版本号修改，则需要重新编译下相关代码并修订相关代码。每个功能我都会在examples中提供示例，出问题的地方可以从示例处查看使用方法。
- **v2为次版本号**。局部的变动，包括功能架构的微调、更多功能的增加等。使用新版本也要重新编译下相关代码进行验证。
- **v3为修订版本号**。以修复Bug为主，新增加的函数将在v2发布中进行说明。
- **v4为希腊字母版本号**。在本项目中，使用三种希腊字母，分别为alpha、beta、RC。正式发布的版本将不会包含v4部分。
  - **alpha**：测试版本，以实现软件功能为主，因此存在较多bug，需要持续修改。
  - **beta**：相对于alpha版本已经修复了大量的bug，但仍需要大量测试。
  - **RC(Release Candidate)**：该版本功能不再增加，和最终发布版功能一样。这个版本有点像最终发行版之前的一个类似预览版，稳定无误后直接发布这个的发布就标明离最终发行版不远了。

# 注意事项
下面这些注意事项是当前并未支持或者支持不好的功能，后续都会进行优化。

- 版本问题。v1.v2.v3.....，
  - 如果v1没改，则API操作无变化，各位可方向使用。若v1改动，则某些API操作发生变化，需要编译下修订下相关代码。
  - v2的改动，仅表示修复了bug+增加了一些显著的新功能。

- BPU操作相关。
  - ※模型部署不支持NV12。NV12目前没有做模型检查，使用时候注意下，NV12这种带有压缩的数据需要研究清楚，对齐、拷贝之后，再补充功能。
  - 模型输出/输出仅支持4维(BPU目前也仅支持4维)。其他维度使用会存在未知问题，部分地方没做想相关的校验。
  - 数据排布由当前库WDR实现。不会通过指定alignshape=validshape来要求BPU做数据对齐，后续会优化内存对齐效率，并支持多维矩阵内存对齐。
  - 少用HB_CHECK_SUCCESS宏定义。HB_CHECK_SUCCESS用于检查BPU一些函数的有效性，后面想办法优化，尽量不定义新的宏定义。

# 优化计划
未来会根据当前需求来支持一些新功能的继承，一般会每个月优化一次。下面这些是初步计划，后面在开发中不断细化这些需求
- BPU部分
  - NV12支持
  - 不定维Tensor的维护
- DNN部分
  - 增加更多后处理函数
- Core部分
  - 增加更多预处理算子
- 更多嵌入式设备引入
  - 集成微雪机器狗的控制
  - 集成tello无人机的控制
- 更多嵌入式功能引入
  - 基于Wifi的数传模块。
- 增加CircularProc模块：圆结构处理相关，包含检测，测量，合作靶标检测相关。
- SLAM功能开发
  - Lidar SLAM中相关函数的集成

# 代码贡献
欢迎各位多多使用，提提建议，目前代码贡献规范还没建立起来，有问题就直接在issue中提吧，解决不了就把项目发出来调试看看。