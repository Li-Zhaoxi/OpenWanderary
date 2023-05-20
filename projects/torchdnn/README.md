
torchdnn：用于训练网络，博客中的一些内容都会记录在这里。
该部分参考了代码：https://github.com/hust-linyi/MedISeg


训练UNet
python train.py --task baseline --fold 0 --train-gpus 0 --dataset=2DMRACerebrovascular --dataroot="D:/01 - datasets/008 - 2DMRACerebrovascular" --resultroot="D:/01 - datasets/008 - 2DMRACerebrovascular/Experiments" --train-batch-size=8 --train-workers=4 --name=unet

python test.py --task baseline --fold 0 --train-gpus 0 --dataset=2DMRACerebrovascular --dataroot="D:/01 - datasets/008 - 2DMRACerebrovascular" --resultroot="D:/01 - datasets/008 - 2DMRACerebrovascular/Experiments" --test-save-flag=true --name=unet