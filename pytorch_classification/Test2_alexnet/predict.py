import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)       #预处理时自动将channel换到第一个维度
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)       #添加batch维度

    # read class_indict
    json_path = './class_indices.json'      #读取类别文件
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)     #解码成需要的字典

    # create model
    model = AlexNet(num_classes=5).to(device)       #初始化

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))     #载入网络模型

    model.eval()        #进入eval，关闭掉dropout方法
    with torch.no_grad():       #让pytroch不去跟踪损失梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()     #正向传播，通过squeeze将batch维度压缩掉
        predict = torch.softmax(output, dim=0)      #通过softmax将输出变为概率分布
        predict_cla = torch.argmax(predict).numpy()     #通过argmax获得概率最大处的索引值

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
