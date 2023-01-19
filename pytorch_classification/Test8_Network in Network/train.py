import os
import json
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import NiN


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
 
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(size=224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
 
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # os.getcwd()获取当前文件所在的目录，两个点表示返回上一层目录，os.path.join将后面两个路径连起来
    image_path = os.path.join(data_root, "data_set1", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),      #加载数据集，训练集
                                         transform=data_transform["train"])           #数据预处理
    train_num = len(train_dataset)                                                    #通过len函数打印训练集有多少张图片
 
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx        #获取每一种类别对应的索引
    cla_dict = dict((val, key) for key, val in flower_list.items())     #遍历刚刚获得的字典，将key和val反过来
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)      # 将cla_dict编码为json格式
    with open('class_indices.json', 'w') as json_file:      #方便预测时读取信息
        json_file.write(json_str)
 
    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
 
    train_loader = torch.utils.data.DataLoader(train_dataset,       #加载数据
                                               batch_size=batch_size, 
                                               shuffle=True,    #随机数据
                                               num_workers=nw)      #线程个数，windows为0
 
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),       #载入测试集
                                            transform=data_transform["val"])        #预处理函数
    val_num = len(validate_dataset)     #统计测试集的文件个数
    validate_loader = torch.utils.data.DataLoader(validate_dataset,     #加载数据
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)
 
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = NiN(num_classes=5, init_weights=True)
    net.to(device) 
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    
    epochs = 10
    save_path = './NIN.pth'
    best_acc = 0.0          #最佳准确率
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()     #只希望在训练过程中随机失活参数，所以通过net.train()和net.eval()管理dropout方法，这样还可以管理BN层
        running_loss = 0.0      #统计训练过程中的平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):     #遍历数据集
            images, labels = data       #分为图像和标签
            optimizer.zero_grad()       #清空之前的梯度信息
            outputs = net(images.to(device))        #正向传播，指定设备
            loss = loss_function(outputs, labels.to(device))        #计算预测值和真实值的损失
            loss.backward()     #反向传播到每一个节点中
            optimizer.step()        #更新每一个节点的参数
 
            # print statistics
            running_loss += loss.item()     #累加loss
 
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
 
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():       #禁止pytroch对参数进行跟踪，在验证时不会计算损失梯度
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:        #遍历验证集
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]        #输出最大值设置为预测值
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()      #预测标签和真实标签对比，计算预测正确的个数
 
        val_accurate = acc / val_num        #测试集的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
 
        if val_accurate > best_acc:     #当前的大于历史最优的
            best_acc = val_accurate     #赋值
            torch.save(net.state_dict(), save_path)     #保存权重
 


if __name__ == '__main__':
    main()

