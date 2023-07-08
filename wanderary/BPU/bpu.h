#ifndef WDR_BPU_HPP_
#define WDR_BPU_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <unordered_map>
#include <memory>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/gapi/s11n.hpp>

#define HB_CHECK_SUCCESS(value, errmsg)                    \
  {                                                        \
    /*value can be call of function*/                      \
    auto ret_code = value;                                 \
    if (ret_code != 0)                                     \
    {                                                      \
      LOG(ERROR) << errmsg << ", error code:" << ret_code; \
      abort();                                             \
    }                                                      \
  }

namespace wdr
{

  namespace BPU
  {
    /////////////// Basic Definitions //////////////////
    class BpuNets;
    class BpuMats;

    enum NETIO
    {
      NET_INPUT = 0,
      NET_OUTPUT = 1
    };

    enum DEVICE
    {
      NET_CPU = 0,
      NET_BPU = 1
    };

    ////////////// 记录一个"连续的Tensor属性"序列 //////////////
    struct NetIOInfo
    {
      std::vector<hbDNNTensorProperties> infos;

      inline bool valid(int idx) const;
      inline int size() const;
      inline const hbDNNTensorProperties &operator[](int idx) const;
    };

    inline bool NetIOInfo::valid(int idx) const
    {
      if (idx >= 0 && idx < infos.size())
        return true;
      else
        return false;
    }

    inline int NetIOInfo::size() const
    {
      return infos.size();
    }

    inline const hbDNNTensorProperties &NetIOInfo::operator[](int idx) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + "is valid, total number is " + std::to_string(this->size()));
      return infos[idx];
    }

    //////////////  记录一个模型的IO Tensor属性信息 /////////////////
    struct NetInfos
    {
      std::string modelname;
      NetIOInfo input_infos;
      NetIOInfo output_infos;

      inline const NetIOInfo &operator[](NETIO mode) const;
    };

    inline const NetIOInfo &NetInfos::operator[](NETIO mode) const
    {
      if (mode == NETIO::NET_INPUT)
        return input_infos;
      else if (mode == NETIO::NET_OUTPUT)
        return output_infos;
      else
        CV_Error(cv::Error::StsOutOfRange, "Invalid mode value: " + std::to_string(int(mode)));
      return input_infos; // 这里永远不会被执行
    }

    // 考虑到MatSize的指针做不到内存管理，交给用户操作风险太大。因此考虑到MatShape实际上是vector<int>，因此这里做了一些扩展
    // 后续考虑派生C语言接口的属性类
    class TensorSize
    {
    public:
      TensorSize() {}
      ~TensorSize() {}
      inline void create(const std::vector<int> &_shapes); // 利用一段vector创建
      inline void create(int dimnum, const int *p);
      inline int dims() const;
      inline const int &operator[](int i) const;
      inline int &operator[](int i);
      inline void copyTo(std::vector<int> &_shapes) const;
      bool operator==(const TensorSize &tz) const;
      inline bool operator!=(const TensorSize &tz) const;
      bool operator<=(const TensorSize &tz) const; // 判断shape是否都<=目标shape
      bool operator>=(const TensorSize &tz) const; // 判断shape是否都>=目标shape
      //
      inline void clear();
      inline void push_back(int dim);
      inline void insert(int pos, int val);
      //

    private:
      std::vector<int> shapes;
    };

    inline void TensorSize::create(const std::vector<int> &_shapes)
    {
      shapes = _shapes;
    }

    inline void TensorSize::create(int dimnum, const int *p)
    {
      shapes.resize(dimnum);
      for (int k = 0; k < dimnum; k++)
        shapes[k] = p[k];
    }

    inline int TensorSize::dims() const
    {
      return shapes.size();
    }

    inline const int &TensorSize::operator[](int i) const
    {
      if (i < 0 || i >= dims())
      {
        std::stringstream ss;
        ss << "Invalid index: " << i << ", max dim: " << dims();
        CV_Error(cv::Error::StsAssert, ss.str());
      }
      return shapes[i];
    }

    inline int &TensorSize::operator[](int i)
    {
      if (i < 0 || i >= dims())
      {
        std::stringstream ss;
        ss << "Invalid index: " << i << ", max dim: " << dims();
        CV_Error(cv::Error::StsAssert, ss.str());
      }
      return shapes[i];
    }

    inline bool TensorSize::operator!=(const TensorSize &tz) const
    {
      return !this->operator==(tz);
    }

    inline void TensorSize::push_back(int dim)
    {
      shapes.push_back(dim);
    }

    inline void TensorSize::clear()
    {
      shapes.clear();
    }

    inline void TensorSize::insert(int pos, int val)
    {
      shapes.insert(shapes.begin() + pos, val);
    }

    inline void TensorSize::copyTo(std::vector<int> &_shapes) const
    {
      _shapes = shapes;
    }
    ////////////// Tensor数据交互管理器 /////////////////
    // 拿到的数据就是已经分配好的了，目前矩阵只支持4维
    class BpuMat
    {
    public:
      BpuMat() {}
      ~BpuMat() {}

      // // 基本信息输出
      bool empty() const;                                              // 数据是否为空
      int batchsize(bool aligned = false) const;                       // 返回Batchsize
      int channels(bool aligned = false) const;                        // 返回通道数
      cv::Size size(bool aligned = false) const;                       // 返回维度
      int total(bool aligned = false) const;                           // 返回元素总数
      size_t elemSize() const;                                         // 返回每个元素的字节数，不要压缩
      void shape(TensorSize &tensorshape, bool aligned = false) const; // 返回Tensor尺寸
      // // 元素操作，直接操作原始数据，
      // 有大批量操作的最好是调取原始指针，或者转为cv::Mat去操作
      // 调用at是方便少量数据的IO，检查项较多，建议少批量使用
      // 安全起见，每次调用都是重新获取Tensor的数据指针。
      template <typename _Tp>
      _Tp &at(int ib, int ic, int ih, int iw);
      template <typename _Tp>
      const _Tp &at(int ib, int ic, int ih, int iw) const;
      template <typename _Tp>
      _Tp *data();
      /////// cv::Mat数据拷贝到Tensor中
      // 一种是正常通道矩阵，如果检查到维度与aligned不匹配，则数据对齐由BPU处理
      // 数据拷入，明确基本需求，用户不会花费大量时间输入，可以就按照图像输入
      // 推理也不一定是图像数据，Padding交给代码自动处理，
      // 如果w*h*c存在，就是自动转换格式，如果不存在，就是原始拷贝格式，如果不考虑padding，则由BPU自动来处理排布
      void copyFrom(cv::InputArray cvmat);
      inline void operator<<(cv::InputArray cvmat); // 数据拷贝到Tensor，只能调用一次<<

      // 通过重载<<和>>完成矩阵的赋值
      // 这里拼好的矩阵可以直接通过<<的方式进行赋值，箭头方向指的是赋值方向
      // 数据拷出，明确基本需求
      // 1. 拷贝出原始数据，推理数据还需要后处理，因此没必要补充带有Mask的拷贝，convertTo需求不强，有需求可以转为CVmat处理
      // 2. 拷贝出带有Mask的数据，如果不考虑padding问题，则由WDR来处理数据排布
      void copyTo(cv::OutputArray cvmat, bool aligned = false) const;
      inline void operator>>(cv::OutputArray cvmat) const; // Tensor数据拷出，只能调用一次>>

      // 输入：Padding推断，若输入是8UC3，则默认是BGR，会根据输入自己做变换。
      // 若输入是图像，则会自动做处理，否则需要人工处理
      // P << Img << RGB2BGR << cv::Size();

      // 输入一定与某个shape对齐，如果自己做好了padding，则一定有一个是对齐的

      // 输出：Padding无法推断，可以利用enum进行推断

    private:
      friend class BpuMats;
      void update();
      std::vector<int> validdims, aligneddims;
      int32_t alignedByteSize{0}, tensorLayout{0};

    private:
      int idxtensor{-1};
      std::shared_ptr<NetIOInfo> properties{nullptr};
      std::shared_ptr<std::vector<hbDNNTensor>> matset{nullptr};
    };

    template <typename _Tp>
    _Tp &BpuMat::at(int ib, int ic, int ih, int iw)
    {
      CV_Assert(ib >= 0 && ib < aligneddims[0]);
      CV_Assert(ic >= 0 && ic < channels(true));
      cv::Size wh = this->size(true);
      CV_Assert(ih >= 0 && ih < wh.height && iw >= 0 && iw < wh.width);
      CV_Assert(sizeof(_Tp) <= elemSize());

      int idx = 0;
      if (tensorLayout == HB_DNN_LAYOUT_NCHW)
        idx = aligneddims[3] * (aligneddims[2] * (ib * aligneddims[1] + ic) + ih) + iw;
      else
        idx = aligneddims[3] * (aligneddims[2] * (ib * aligneddims[1] + ih) + iw) + ic;

      return *(((_Tp *)matset->at(idxtensor).sysMem[0].virAddr) + idx);
    }

    template <typename _Tp>
    const _Tp &BpuMat::at(int ib, int ic, int ih, int iw) const
    {
      CV_Assert(ib >= 0 && ib < aligneddims[0]);
      CV_Assert(ic >= 0 && ic < channels(true));
      cv::Size wh = size(true);
      CV_Assert(ih >= 0 && ih < wh.height && iw >= 0 && iw < wh.width);
      CV_Assert(sizeof(_Tp) <= elemSize());

      int idx = 0;
      if (tensorLayout == HB_DNN_LAYOUT_NCHW)
        idx = aligneddims[3] * (aligneddims[2] * (ib * aligneddims[1] + ic) + ih) + iw;
      else
        idx = aligneddims[3] * (aligneddims[2] * (ib * aligneddims[1] + ih) + iw) + ic;

      return *(((const _Tp *)matset->at(idxtensor).sysMem[0].virAddr) + idx);
    }

    template <typename _Tp>
    _Tp *BpuMat::data()
    {
      return (_Tp *)matset->at(idxtensor).sysMem[0].virAddr;
    }

    inline void BpuMat::operator<<(cv::InputArray cvmat)
    {
      copyFrom(cvmat);
    }

    inline void BpuMat::operator>>(cv::OutputArray cvmat) const
    {
      copyTo(cvmat, false);
    }

    //////////////  模型输入管理器 【非线程安全】 /////////////////
    class BpuMats
    {
    public:
      ///// 创建销毁相关，不提供给用户主动release的过程，在BpuMats的生命周期结束后自动释放
      BpuMats();
      ~BpuMats();

      inline int size() const; // 返回Tensor矩阵的个数

      // 内存交换，CPU<->BPU拷贝
      // 内存状态检查：
      // 检测到矩阵有数据输入，则更换CPU模式
      // 调用forward之后，
      // 检测到矩阵有数据输出，则确定模式为CPU更换为CPU模式
      // 参考上车下车模式，
      // 若数据被输入，则输入后，模式更换为cpu
      // 若数据被输出，则输出后，
      void bpu();                   // 数据移到BPU
      void cpu();                   // 数据移到CPU
      inline DEVICE device() const; // 返回当前设备名

      // 返回Tenosr序列的子集，共享参数，有时候网络A的输入是网络B的连续一段
      BpuMats operator()(cv::Range &_range) const;

      //// 单Tensor返回，只保留矩阵数值的IO，不要留分配
      BpuMat operator[](int idx) const;

    private: // 友元接口相关
      friend class BpuNets;
      void release();                      // 初始化
      void create(const NetIOInfo &infos, bool autopadding); // 分配内存

    private:
      cv::Range range;
      // 这里使用智能指针，如果有共享，这部分永远不会被释放
      // 有深拷贝需求，在BPUMat自己做好适配
      std::shared_ptr<DEVICE> dev{nullptr};
      std::shared_ptr<NetIOInfo> properties{nullptr};
      std::shared_ptr<std::vector<hbDNNTensor>> matset{nullptr};
    };

    inline int BpuMats::size() const
    {
      return range.end - range.start;
    }

    inline DEVICE BpuMats::device() const
    {
      CV_Assert(dev != nullptr);
      return *dev;
    }

    // BpuMats：操作一批Tenosr，保证内存的连续性，所依赖的功能
    // 2. BPU参数的赋值通过BPUMat处理

    ///// 下面定义的NetIOInfo，NETIO，NetInfos是用于查看模型信息专用的 用于打印模型信息专用，

    // 读取网络就把所有的参数读取出来，

    // Bpu网络数据结构设计，不要太多函数，能
    // 1. 指针接口不能暴漏
    // 2. 不要将分配释放的工作交给用户
    // 3. 不要设计的过于复杂，有特殊需求就面向过程，对于开发者来说，就应该是面向对象的
    // 4. 给面向过程的用户留出对应的函数
    class BpuNets
    {
    public:
      BpuNets();
      ~BpuNets();
      void readNets(const std::vector<std::string> &modelpaths);
      void release();

      /////////////////// 基本信息操作

      // 返回模型个数
      inline int total() const;

      // 根据模型名返回索引
      int name2index(const std::string &modelname) const;

      // 根据索引返回模型名
      const std::string &index2name(int idx) const;

      // 检查索引有效性
      inline bool valid(int idx) const;

      // 打印模型信息。因此需要补充两个功能，函数at和重载[]，返回目标索引，注意：都是常引用返回，不允许修改其中的值
      // 此外，模型信息需要支持std::cout << 这种功能，因此相关的功能在函数外重载
      // Tensor的细节调用函数分配，有计算需求，转为OpenCVMat即可，没必要开发重复功能
      // 这种方式耗时，不要频繁调用
      inline const NetInfos &at(int idx) const;
      inline const NetInfos &operator[](int idx) const;

      /////////////////// BPU核心功能：分配+推理
      // 输入输入模型索引，检查Tensor序列属性的一致性，检查项：tensor个数+数据类型+数据尺寸+数字字节
      bool checkTensorProperties(int idx, const BpuMats &bpumats, bool input, std::string &errmsg) const;

      // 输入模型索引，分配输入输出的数据，存在BpuMats中
      void init(int idx, BpuMats &input_mats, BpuMats &output_mats, bool autopadding) const;

      // 输入模型索引+输入输出Tensor，完成推理
      void forward(int idx, const BpuMats &input_mats, BpuMats &output_mats) const;

    private:
      hbPackedDNNHandle_t pPackedNets{nullptr};
      std::vector<std::pair<std::string, hbDNNHandle_t>> netsMap;
      std::vector<NetInfos> netinfos;
    };

    inline int BpuNets::total() const
    {
      return netsMap.size();
    }

    inline bool BpuNets::valid(int idx) const
    {
      if (idx >= this->total() || idx < 0)
        return false;
      else
        return true;
    }

    inline const NetInfos &BpuNets::at(int idx) const
    {
      if (!this->valid(idx))
        CV_Error(cv::Error::StsOutOfRange, std::to_string(idx) + "is valid, total number is " + std::to_string(this->total()));

      return netinfos[idx];
    }

    inline const NetInfos &BpuNets::operator[](int idx) const
    {
      return this->at(idx);
    }

    // 加载网络
    void readNets(const std::vector<std::string> &modelpaths,
                  hbPackedDNNHandle_t &pPackedNets,
                  std::unordered_map<std::string, hbDNNHandle_t> &netsMap);
    // 释放网络
    void releaseNets(hbPackedDNNHandle_t &pPackedNets);
    // 读取网络输入输出属性
    void readNetProperties(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensorProperties> &properties, bool input);
    // 加载网络Tensor的尺寸，若aligned=true，则获取的是alignedshape的尺寸，否则获取的是validshape的尺寸
    void shape(const hbDNNTensorProperties &property, TensorSize &tensorshape, bool aligned = false);
    void shape(cv::InputArray src, TensorSize &cvshape);
    // 内存分配
    void createTensors(const std::vector<hbDNNTensorProperties> &properties, std::vector<hbDNNTensor> &tensors, bool autopadding = true);
    void createTensors(const hbDNNHandle_t dnn_handle, std::vector<hbDNNTensor> &tensors, bool input, bool autopadding = true);
    void createTensors(const hbDNNTensorProperties &property, hbDNNTensor &bputensor);
    // 内存释放
    void releaseTensors(std::vector<hbDNNTensor> &tensors);
    // 内存刷新，若upload=true,则为CPU刷新到BPU上，否则为BPU刷新到CPU上
    void flushBPU(hbDNNTensor &dst, bool upload);
    // 内存对齐，不支持NV12，仅支持4维矩阵，后面针对这两个问题再优化+效率优化
    void alignMemory(const unsigned char *src, const TensorSize &srcshape, unsigned char *dst, TensorSize &dstshape, int elementsize);
    // 内存拷贝：数据拷贝到Tensor中
    // cv::Mat转Tensor，用于转换输入的Mat到Tensor中
    void bpuMemcpy(hbDNNTensor &dst, const uint8_t *src, int memsize = -1, bool flush = true);
    void bpuMemcpy(cv::InputArray src, hbDNNTensor &dst, bool flush = true);
    // 内存拷贝：数据从Tensor拷贝到CPU数据中
    void bpuMemcpy(uint8_t *dst, hbDNNTensor &src, int memsize = -1, bool flush = true);
    void bpuMemcpy(hbDNNTensor &src, cv::OutputArray dst, bool align = true, bool flush = true);
    // 网络推理，两种模式，vector自动保证内存连续，而指针的方式需要时候需要自己注意下内存连续问题。
    void forward(const hbDNNHandle_t dnn_handle, const std::vector<hbDNNTensor> &inTensors, std::vector<hbDNNTensor> &outTensors, int waiting_time = 0);
    void forward(const hbDNNHandle_t dnn_handle, const hbDNNTensor *_inTensors, hbDNNTensor *_outTensors, int waiting_time = 0);

    // 指针这种接口不能暴漏出来

    // 网络推理认定为算法，因此基于Ptr模式进行创建
    // 单模型单算法推理拒绝冲突，推理时做好维度检查
    // 思考任务释放模式，通过枷锁的方式管理

    ///////// 数据可视化，下面都是返回一行的string
    // 如果重载的话，很容易出现冲突的问题，因此为了方便可视化
    // 将每种类型返回一个字符串，可以直接std::cout << format**() << std::endl;处理
    // 有些数据返回的是一个字符串，有的返回的是一个tabel，试用时候可以注意下。

    // 下面这些是返回一行字符串的
    std::string formathbDNNQuantiShift(const hbDNNQuantiShift &c1);
    std::string formathbDNNTensorShape(const hbDNNTensorShape &c1);
    std::string formathbDNNQuantiScale(const hbDNNQuantiScale &c1);
    std::string formathbDNNQuantiType(const hbDNNQuantiType &c1);
    std::string formathbDNNTensorLayout(const hbDNNTensorLayout &c1);
    std::string formathbDNNDataType(const hbDNNDataType &c1);

    // 下面这些是返回一批信息的，这里借用Json的序列化方式
    std::string formathbDNNTensorProperties(const hbDNNTensorProperties &c1);
    std::ostream &operator<<(std::ostream &out, const NetIOInfo &c1);  // 打印一组输入/输出的Tensor信息
    std::ostream &operator<<(std::ostream &out, const NetInfos &c1);   // 打印一个模型的所有Tensor信息
    std::ostream &operator<<(std::ostream &out, const TensorSize &ts); // 打印形状参数信息
  }                                                                    // end BPU

} // end wdr

// // hbDNNTensorProperties没有定义在wdr中因此需要
std::ostream &operator<<(std::ostream &out, const hbDNNTensorProperties &c1);

#endif