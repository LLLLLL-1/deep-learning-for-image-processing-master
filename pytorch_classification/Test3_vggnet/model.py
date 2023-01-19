import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    #传入提取特征网络结构，init_weights表示是否对权重进行初始化
    def __init__(self, features, num_classes=1000, init_weights=False):     
        super(VGG, self).__init__()
        self.features = features

        #全连接层分类，提取特征得到的时7*7*512，要先进行展平处理才能进行分类
        self.classifier = nn.Sequential(        
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        #是否对网络参数初始化
        if init_weights:
            self._initialize_weights()

    #正向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7，展平操作，从第一个维度开始
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    #初始化参数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 卷积层提取特征
def make_features(cfg: list):       #提取特征的函数，使用时只需要传入对应的列表
    layers = []     #定义一个空列表存放我们定义的每一层结构
    in_channels = 3     #彩色图像，channel为3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0...
    for v in cfg:       #遍历配置列表
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]       #创建最大池化下采样层 
        else:       #卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)        #第一层时彩色图像in_channels=3，v=64个卷积核
            layers += [conv2d, nn.ReLU(True)]       #将刚刚定义的卷积层和Relu激活函数拼接在一起，添加到layers列表中
            in_channels = v     #特征矩阵经过该层卷积之后，输出的深度变成v
    return nn.Sequential(*layers)       #将列表通过非关键字的形式传入


#定义一个字典文件，字典的每个key代表一个模型的配置文件
cfgs = {        
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


#实例化vgg网络
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]      #得到cfg

    model = VGG(make_features(cfg), **kwargs)       #**kwargs可变长度的字典变量
    return model
