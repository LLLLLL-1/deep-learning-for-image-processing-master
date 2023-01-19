import torch
from torch import nn




# class GlobalAvlPool2d(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return nn.functional.avg_pool2d(x, (x.shape[2], x.shape[3]))

class NiN(nn.Module):
    def __init__(self,num_classes,init_weights=False):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.NiN_block(3, 96, 11, 4, 2),                 # input[3, 224, 224]  # output[96, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.NiN_block(96, 256, 5, 1, 2),                # output[256, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.NiN_block(256, 384, 3, 1, 1),               # output[284, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            self.NiN_block(384, num_classes, 3, 1, 1),                
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        if init_weights:                                            #初始化权重，当初始化时设置为true，就会使用这个函数
            self._initialize_weights()

    
    def NiN_block(self,in_channels, out_channels, kernel_size, stride, padding):
        nin = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.ReLU()
        )
        return nin

    def forward(self,x):
        return self.net(x)


    def _initialize_weights(self):
        for m in self.modules():                                    #遍历self.modules模块，继承自nn.Module，会遍历我们定义的每一个层结构
            if isinstance(m, nn.Conv2d):                            #isinstance函数用来比较得到的层结构是否等于给定的类型，是卷积层时，则使用凯明初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                          #如果传进来的是全连接层
                nn.init.normal_(m.weight, 0, 0.01)                  #通过正态分布对权重赋值，均值为0，方差为0.01
                nn.init.constant_(m.bias, 0) 

    def test_output_shape(self):
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.net:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)

    


# net = NiN()
# print(net)
