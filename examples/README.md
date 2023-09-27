


# BPU示例

## print-infos: 模型信息可视化
提供了class和api操作的模式，方便理解如何使用封装的Class和APIs

## validate-infer

```
./build/examples/BPU/validate-infer --dataroot=projects/torchdnn/data/dcmt/debug_infer --filelist=projects/torchdnn/data/dcmt/debug_infer/filelists.txt --binpath=projects/torchdnn/data/dcmt/DCMT.bin --modelname=DCMT --dtypes=uint8,uint8,float32,float32,float32
```


# MIPI 读取示例


BPU部署校验器

1. 初始化
2. 预处理
3. 后处理
4. 


1. 预处理后处理没必要统一，尽可能封装函数，把函数测试稳定即可。debug时候学会用gbd调试
2. 完成跟踪的demo主流程。