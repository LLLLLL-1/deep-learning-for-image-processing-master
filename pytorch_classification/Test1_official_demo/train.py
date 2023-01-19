import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms




def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)        #iter将val_loader变为可迭代的迭代器
    val_image, val_label = next(val_data_iter)   #next可得到一批数据，即图像和标签
    
    # classes = ('plane', 'car', 'bird', 'cat',       #导入标签，元组类型（值不能改变）
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()                                                         #实例化模型
    loss_function = nn.CrossEntropyLoss()                                 #定义损失函数，这个函数已经包含softmax函数，所以网络的最后一层不再使用
    optimizer = optim.Adam(net.parameters(), lr=0.001)                    #定义优化器，Adam优化器，第一个参数是需要训练的参数，将lenet可训练的参数都进行训练，lr是学习率

    for epoch in range(5):  # loop over the dataset multiple times        #将训练集训练多少次

        running_loss = 0.0                                                #累加训练过程中的损失
        for step, data in enumerate(train_loader, start=0):               #通过循环遍历训练集样本，通过enumerate函数，不仅返回数据，还可以返回每一批数据的步数
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data                                         #将数据分成输入和标签

            # zero the parameter gradients
            optimizer.zero_grad()                                         #将历史损失梯度清零
            # forward + backward + optimize
            outputs = net(inputs)                                         #将输入的图片进行正向传播
            loss = loss_function(outputs, labels)                         #outputs是网络的预测值，labels是输入图片对应的真实标签
            loss.backward()                                               #反向传播
            optimizer.step()                                              #参数的更新

            # print statistics
            running_loss += loss.item()                                   #将损失累加
            if step % 500 == 499:    # print every 500 mini-batches       #每隔500步打印一次数据信息
                with torch.no_grad():                                     #with是一个上下文管理器，torch.no_grad表示在接下来不要计算误差损失梯度
                    outputs = net(val_image)  # [batch, 10]               #进行正向传播
                    predict_y = torch.max(outputs, dim=1)[1]              #寻找最大的index在哪里，[batch, 10] 第0个维度是batch，第一个维度是输出，[1]是索引，得到最大值对应的标签类别
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                                                                          #预测标签类别和真实标签类别比较，在相同的地方返回1  true，.sum()计算在预测过程中预测对了多少样本，得到结果是tensor，通过.item()拿到数值，在除以训练样本的数目
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                                                                          #%d=epoch，%5d是某一轮的多少步， %.3f训练过程中累加的误差
                    running_loss = 0.0                                    #清零，进行下一次的500步

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)                               #保存模型


if __name__ == '__main__':
    main()
