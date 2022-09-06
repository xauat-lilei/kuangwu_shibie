import os
import json

import torch
import time
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
classes=['bantong', 'chensha', 'chitie', 'cihuangtie','citie','dusha','fangjie','fangqian','ganlanshi','getie','heiwu','hetie','huangtie','huangtong',
         'huimu','huiti','huitong','kongqueshi','lantong','lvtu','ruanmeng','shanxin','shiying','tiemu','xionghuang','yingshi']
import model
import matplotlib.pyplot as plt
import numpy as np
from huatu import imshow

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = data_root +"/data/" # kuangshi data set path
    train_dataset = datasets.ImageFolder(root=image_path+ "/train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    kuangshi_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in kuangshi_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=25)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test结果
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    net = model.VGG(img_size=224, input_channel=3, num_class=26)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 500
    best_acc = 0.0

    train_acc_list, val_accurate_list = [], []
    train_loss_list, val_loss_list = [], []
    time_list, epoch_list = [], []
    for epoch in range(epochs):
        # train
        # train
        start = time.time()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_accurate = 0
        train_bar = tqdm(train_loader)
        net.train()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            probs, pred_y = outputs.data.max(dim=1)  # 得到概率
            train_acc += (pred_y == labels.to(device)).sum() / labels.size(0)

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(epoch + 1, epochs, int(rate * 100), a, b), end='')

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc * 100 / len(train_loader)
        #     print('train_loss:{:.3f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # validate
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for step, data in enumerate(val_bar):
                val_images, val_labels = data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels)
                val_loss += loss.data
                probs, pred_y = outputs.data.max(dim=1)  # 得到概率
                val_accurate += (pred_y == val_labels.to(device)).sum() / val_labels.size(0)
                rate = (step + 1) / len(validate_loader)
                a = "*" * int(rate * 50)
                b = "." * (50 - int(rate * 50))
                print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(epoch + 1, epochs, int(rate * 100), a, b),
                      end='')
        val_loss = val_loss / len(validate_loader)
        val_accurate = val_accurate * 100 / len(validate_loader)
        val_loss_list.append(val_loss)
        val_accurate_list.append(val_accurate)
        end = time.time()
        print(
            ' epoch[{}/{}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}'.format(
                epoch + 1, epochs, train_loss, train_acc), end='')
        print(
            ' epoch[{}/{}]  Validate Loss:{:>.6f}  Validate Acc:{:>3.2f}'.format(
                epoch + 1, epochs, val_loss, val_accurate), end='')
        time_ = int(end - start)
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 / 60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % (m, s)
        time_list.append(time_str)
        # 打印所用时间
        print(time_str)
        # 添加epoch_list
        epoch_list.append(epoch)

        # 保存数据
    import random
    import pandas as pd
    from datetime import datetime
    # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df1 = pd.DataFrame(columns=['time_str', 'epoch', 'train_acc', 'val_accurate'])  # 列名
    df1.to_csv("E:\\李雷\\矿石图像训练\\VGG\\acc.csv", index=False)  # 路径可以根据需要更改

    df2 = pd.DataFrame(columns=['time_str', 'epoch', 'train_loss', 'val_loss'])  # 列名
    df2.to_csv("E:\\李雷\\矿石图像训练\\VGG\\loss.csv", index=False)  # 路径可以根据需要更改
    # 将数据保存在一维列表
    # 保存准确率数据
    list1 = [time_str, epoch_list, train_acc_list, val_accurate_list]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data1 = pd.DataFrame([list1])
    data1.to_csv('E:\\李雷\\矿石图像训练\\VGG\\acc.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

    # 保存LOSS数据
    list2 = [time_str, epoch_list, train_loss_list, val_loss_list]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data2 = pd.DataFrame([list2])
    data2.to_csv('E:\\李雷\\矿石图像训练\\VGG\\loss.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

    ###################################################################
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['val_accurate'] = val_accurate_list
    Loss['train_loss'] = train_loss_list
    Loss['val_loss'] = val_loss_list
    print('Finished Training')
    import huatu
    huatu.plot_history(epochs, Acc, Loss)
    # 展示结果
    correct = 0
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images_ = images
    images_ = images_.to(device)
    labels = labels.to(device)

    test_output = net(images_)
    test_preds = torch.max(test_output, 1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    fig = plt.figure(figsize=(12, 12))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("{} ,({})".format(classes[test_preds[idx].item()], classes[labels[idx].item()]),
                     color=("green" if test_preds[idx].item() == labels[idx].item() else "red"))


if __name__ == '__main__':
    main()
