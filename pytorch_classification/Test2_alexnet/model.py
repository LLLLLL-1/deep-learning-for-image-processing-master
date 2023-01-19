import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(                              #将一系列的层结构打包，形成一个新的结构，取名features，用于专门提取图像特征，对比之前LeNet可以精简一些代码
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(                            #包含最后面的3层全连接层，是一个分类器，将全连接层打包为新的结构
            nn.Dropout(p=0.5),                                      #p为失活的比例，默认为0.5
            nn.Linear(128 * 6 * 6, 2048),                           #因为这里搭建网络时使用原文一半的参数，所以这里为128，节点个数为2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),                           #输出为数据集类别的个数，在初始化时传入的
        )
        if init_weights:                                            #初始化权重，当初始化时设置为true，就会使用这个函数
            self._initialize_weights()                              #在当前版本pytroch在卷积层和全连接层中自动使用凯明初始化方法

    def forward(self, x):                                           #正向传播
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)                           #展平处理，从channel维度进行展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():                                    #遍历self.modules模块，继承自nn.Module，会遍历我们定义的每一个层结构
            if isinstance(m, nn.Conv2d):                            #isinstance函数用来比较得到的层结构是否等于给定的类型，是卷积层时，则使用凯明初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                          #如果传进来的是全连接层
                nn.init.normal_(m.weight, 0, 0.01)                  #通过正态分布对权重赋值，均值为0，方差为0.01
                nn.init.constant_(m.bias, 0)                        #偏置为0
