
开源项目OpenWanderary[wɒndərəri]是一个面向多传感器集成的开源框架，旨在提升基于Linux的嵌入式系统研发与部署效率。项目遵循"从实践中来，到实践中去"的理念，所有功能均源于实际开发需求，并最终服务于嵌入式应用场景。

在嵌入式开发实践中，开发者常面临以下挑战：
- 硬件/设备厂商提供的API存在显著风格差异（如接口命名规范、调用方式等），却实现相似功能，这种异构性导致开发者需要掌握多种实现方式，从而显著增加了学习曲线与测试验证成本。
- 在嵌入式项目开发中，传感器接口、通信协议、基础算法等基础功能模块常需重复实现，虽然这些功能逻辑相对简单，但开发与测试环节平均会消耗项目20%-30%的工时，显著延长研发周期。
- 缺乏标准化中间件阻碍创新效率，采用标准化中间件可使项目周期缩短，而节省的工时可直接转化为更有意义的创新。

为此，OpenWanderary(缩写WDR)基于我个人过去多年的开发实践中的痛点积累，渐进式提供一系列开源中间件库，以缓解上述问题，打造一个大家都喜欢的库。OpenWanderary重点为以下几点，后面新增的功能都是围绕这几点展开：
- **Open** Source。WDR中扩展的库都是完全开源的，提供足够的灵活性。
- Integrate **Wander**ing Sensors/Device。集成多设备/多传感器的相关库，以及使用这些设备的相关算法。
- Libr**ary**。WDR生成的是动态库，拿过来就直接使用，并配套完整示例工程与解决方案。

轮子造一次就足够了，无需重复构造。下面有一些设计设计原则‌：
- **数据类型最小化‌**: 尽可能不构造新的数据类型，降低学习成本。所用数据依赖Eigen/OpenCV/PCL等成熟库，仅当必要且能降低使用成本时引入新类型。
- **硬件抽象层‌**。当前库以Linux为主，如果有些硬件并不需要，通过CMake配置实现平台无关性，按需编译。
- **完全开源**。希望各位使用WDR时以动态链接方式使用，不要直接复制其中的代码到自己的项目中。有优化项需求希望以pull requrest形式更新，欢迎各位贡献代码。
- **提供优质的开发API**。整体代码，以C++17为基础进行开发，开发过程中参考了《Effective C++》中的一些条款。这部分也是一边学习一边优化，优化代码规范，架构规范。


# 一 代码编译

在开始前，请先关注。由于py包安装在~/.local下面，因此需要先配置下环境变量。在`~/.bashrc`后面添加`export PATH=${HOME}/.local/bin:${PATH}`, 然后source一下。

在clone本仓库代码时，项目通过 Git 子模块（Submodules）的形式引入依赖的外部代码库，这些子模块作为独立的版本控制仓库被嵌套在主项目中，以便有效管理复杂项目的模块化依赖关系。为了帮助开发者更好地理解这些子模块的功能定位和使用场景，以下对各子模块进行详细介绍：
- [`3rdparty/indicators`](https://github.com/p-ranav/indicators): 终端进度条库，用于可视化任务进度，辅助代码监控与调试。
- [`3rdparty/mcap`](https://github.com/foxglove/mcap)`&& 3rdparty/mcap_builder`:一种用于多模态日志数据容器文件格式。它支持多个通道的带时间戳的预序列化数据，非常适合在发布/订阅或机器人应用中使用。mcap_builder用于将MCap头文件封装，以便通过find_package找到。
- [`3rdparty/nlohmann_json`](https://github.com/nlohmann/json.git): C++ JSON处理库，提供轻量级JSON解析与生成能力。
- [`3rdparty/pybind11`](https://github.com/pybind/pybind11.git)`&&`[`3rdparty/pybind11_json`](https://github.com/pybind/pybind11_json.git): C++/Python绑定库，实现C++代码到Python模块的封装。。
- [`3rdparty/yaml-cpp`](https://github.com/jbeder/yaml-cpp.git): C++ YAML处理库，支持YAML数据的解析与生成。
- [`3rdparty/waymo-open-dataset`](https://github.com/waymo-research/waymo-open-dataset.git): Waymo开源数据集，提供自动驾驶领域的数据集和工具，用于训练和评估自动驾驶算法。

下面开始我们的源码编译与安装

**Step1: 下载源码**。选择一个合适的目录作为下载路径，然后在终端中执行以下命令，注意`--recursive`参数会‌自动递归克隆所有子模块‌，**一定要确保子模块clone成功**。
```
git clone --recursive https://github.com/Li-Zhaoxi/OpenWanderary.git
cd OpenWanderary
```
当网络环境受限或部分模块clone失败时，可采用分步克隆方式替代单次递归克隆。操作步骤如下：
```
git clone https://github.com/Li-Zhaoxi/OpenWanderary.git
cd OpenWanderary
git submodule update --init --recursive
```

**Step2: 下载依赖数据**。在项目根目录`OpenWanderary`下执行`make download`。OpenWanderary中包含一些依赖数据，在执行单元测试或功能用例时需要用到。

download过程细节如下(`Makefile`)，从地瓜官网下载YOLOv8模型，相关文件保存在`tests/test_data/models`目录下。
```
download:
	bash tests/test_data/models/download_models.sh; #
```

**Step3: 编译/安装依赖项**。在项目根目录`OpenWanderary`下执行`make rely`。
- `requirements.txt`中定义了格式规范化以及pytest的依赖项。
- `build_3rdpary.sh`先通过`sudo apt-get install`安装一些依赖包，然后编译安装前面所介绍的子模块。
  - `git-lfs`: 用于开发者模式，代码仓中有些文件需要通过git-lfs下载/上传。
  - `libcli11-dev`: C++11命令行解析库，用于解析命令行参数。

```
rely:
	set -ex; \
	pip3 install -r requirements.txt -i $(pip_source); \
	pre-commit install; \
	bash build_3rdpary.sh;
```

**Step4: OpenWanderary编译**。在项目根目录`OpenWanderary`下执行`make debug`或`make release`来编译Debug/Release版本的库。

```
debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug; \
	make -j6 -C build/;

release:
	cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release; \
	make -j6 -C build_release/;
```

**Step5: OpenWanderary安装**。参考下面代码可以将OpenWanderary安装到`/usr/local`目录下。
- 头文件安装到`/usr/local/include/wanderary/`。
- 库文件安装到`/usr/local/lib/`，相关库以libwdr_**.so命名。
- cmake文件安装到`/usr/local/lib/cmake/wdr/`。

```
cd OpenWanderary;
cd build_release;
sudo make install;
sudo ldconfig;
```

**Step6: OpenWanderary Python版本安装[可选]**。在步骤4编译完库之后(release为例)，Py包会生成在`build_release/dist`目录下，通过下述指令可完成py包的安装。注意，Py包本质是C++的封装，确保C++库已安装。
```
pip3 install build_release/dist/wanderary-0.1.0-py3-none-any.whl
```

# 二 个人项目嵌入

这里介绍如何在您个人的CMake项目或者Python项目中使用OpenWanderary库，这里提供了一个完整的示例项目[OpenWanderary-examples](https://github.com/Li-Zhaoxi/OpenWanderary-examples)，该示例项目演示了如何在CMake项目中使用OpenWanderary库。

下面是对项目集成的更加详细的说明。

## 2.1 CMake项目配置方案

调用wdr库时仅需要在自己的CMake项目中的CMakeLists.txt中添加如下配置即可，wdr依赖的库会自动被找到并链接。
```
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(wdr REQUIRED)
```

wdr依赖库自动链接的原理是`/usr/local/lib/cmake/wdr/wdr-config.cmake`添加了依赖库的查找，感兴趣可以查看这里的代码。

如果我们需要在自己项目中集成yolo算法并可视化，只需要在target_link_libraries中链接`wdr::wdr_visualization wdr::wdr_apps`即可，相关示例可参考: [examples/CMakeLists.txt](https://github.com/Li-Zhaoxi/OpenWanderary-examples/blob/main/examples/CMakeLists.txt)。

## 2.2 Python项目配置方案

在Python项目中使用wdr库，只需要在Python代码中添加如下代码即可，wdr依赖的库会自动被找到并链接。相关示例可参考：[python/yolo8_image.py](https://github.com/Li-Zhaoxi/OpenWanderary-examples/blob/main/python/yolo8_image.py)

```
import wanderary
```



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
