# 语义分割学习实验-基于VOC数据集
### usage：
1. 下载VOC数据集，将`JPEGImages` `SegmentationClass`两个文件夹放入到data文件夹下。
2. 终端切换到目标目录，运行`python train.py -h`查看训练
```bash
(torch) qust116-jq@qustx-X299-WU8:~/语义分割$ python train.py -h
usage: train.py [-h] [-m {Unet,FCN,Deeplab}] [-g GPU]

choose the model

optional arguments:
  -h, --help            show this help message and exit
  -m {Unet,FCN,Deeplab}, --model {Unet,FCN,Deeplab}
                        输入模型名字
  -g GPU, --gpu GPU     输入所需GPU
```
选择模型和GPU编号进行训练，例如运行`python train.py -m Unet -g 0`

3. 预测需要手动修改`predict.py`中的模型
