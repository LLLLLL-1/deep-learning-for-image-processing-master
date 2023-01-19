import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),           #缩放图片 
         transforms.ToTensor(),                 #转化为tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    #标准化处理

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()                                                           #实例化LeNet
    net.load_state_dict(torch.load('Lenet.pth'))                            #载入权重文件

    im = Image.open('th.jpg')                                                #使用Image模块载入图片
    im = transform(im)  # [C, H, W]，通过PIL载入的图片是高度，宽度，深度，必须变为tensor格式才能进行正向传播，进行预处理，得到[C, H, W]格式
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]，          #使用unsqueeze增加一个维度

    # with torch.no_grad():
    #     outputs = net(im)                                                   #图像传入网络
    #     predict = torch.max(outputs, dim=1)[1].numpy()                      #寻找输出中最大的index
    # print(classes[int(predict)])                                            #传入classes

    with torch.no_grad():
        outputs = net(im)
        predict = torch.softmax(outputs, dim=1)
    print(predict)


if __name__ == '__main__':
    main()
