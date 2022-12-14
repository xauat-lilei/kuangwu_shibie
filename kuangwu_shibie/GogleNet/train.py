import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

classes=['bantong', 'chensha', 'chitie', 'cihuangtie','citie','dusha','fangjie','fangqian','ganlanshi','getie','heiwu','hetie','huangtie','huangtong',
         'huimu','huiti','huitong','kongqueshi','lantong','lvtu','ruanmeng','shanxin','shiying','tiemu','xionghuang','yingshi']

from model import GoogLeNet
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
        'test': transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = data_root+ "/data/"  # kuangshi data set path
    train_dataset = datasets.ImageFolder(root=image_path+"/train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    kuangshi_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in kuangshi_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=25)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size =32
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

    #test??????
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    #????????????
    net = GoogLeNet(num_classes=26, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    import time

    epochs = 200
    best_acc = 0
    save_path = './googleNet.pth'
    train_acc_list, val_accurate_list = [], []
    train_loss_list, val_loss_list = [], []
    for epoch in range(epochs):
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
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()
            outputs = logits
            train_loss += loss.data
            probs, pred_y = outputs.data.max(dim=1)  # ????????????
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
                outputs = net(val_images)
                loss = loss_function(outputs, val_labels)
                val_loss += loss.data
                probs, pred_y = outputs.data.max(dim=1)  # ????????????
                val_accurate += (pred_y == val_labels.to(device)).sum() / val_labels.size(0)
                rate = (step + 1) / len(validate_loader)
                a = "*" * int(rate * 50)
                b = "." * (50 - int(rate * 50))
                print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(epoch + 1, epochs, int(rate * 100), a, b), end='')
        val_loss = val_loss / len(validate_loader)
        val_accurate = val_accurate * 100 / len(validate_loader)
        val_loss_list.append(val_loss)
        val_accurate_list.append(val_accurate)
        end = time.time()
        print(
            ' epoch[{}/{}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}'.format(
                epoch + 1,epochs, train_loss, train_acc), end='')
        print(
            ' epoch[{}/{}]  Validate Loss:{:>.6f}  Validate Acc:{:>3.2f}'.format(
                epoch + 1,epochs, val_loss, val_accurate), end='')
        time_ = int(end - start)
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 / 60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % (m, s)
        # ??????????????????
        print(time_str)
        # ????????????????????????????????????????????????
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    # ????????????
    import random
    import pandas as pd
    from datetime import datetime
    # ??????train_acc.csv???var_acc.csv???????????????loss???accuracy
    df1 = pd.DataFrame(columns=['time_str', 'epoch', 'train_acc', 'val_accurate'])  # ??????
    df1.to_csv("E:\\??????\\??????????????????\\GogleNet\\acc.csv", index=False)  # ??????????????????????????????

    df2 = pd.DataFrame(columns=['time_str', 'epoch', 'train_loss', 'val_loss'])  # ??????
    df2.to_csv("E:\\??????\\??????????????????\\GogleNet\\loss.csv", index=False)  # ??????????????????????????????
    # ??????????????????????????????
    # ?????????????????????
    list1 = [time_str, epochs, train_acc_list, val_accurate_list]
    # ??????DataFrame???Pandas???????????????????????????????????????excel???????????????????????????????????????list?????????????????????????????????DataFrame
    data1 = pd.DataFrame([list1])
    data1.to_csv('E:\\??????\\??????????????????\\GogleNet\\acc.csv', mode='a', header=False, index=False)  # mode??????a,????????????csv?????????????????????

    # ??????LOSS??????
    list2 = [time_str, epochs, train_loss_list, val_loss_list]
    # ??????DataFrame???Pandas???????????????????????????????????????excel???????????????????????????????????????list?????????????????????????????????DataFrame
    data2 = pd.DataFrame([list2])
    data2.to_csv('E:\\??????\\??????????????????\\GogleNet\\loss.csv', mode='a', header=False, index=False)  # mode??????a,????????????csv?????????????????????

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
#????????????
    correct=0
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images_ = images
    images_ = images_.to(device)
    labels = labels.to(device)

    test_output = net(images_)
    test_preds = torch.max(test_output,1)[1].cpu().numpy()
    labels=labels.cpu().numpy()
    fig = plt.figure(figsize=(12,12))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("{} ,({})".format(classes[test_preds[idx].item()], classes[labels[idx].item()]),
                 color = ("green" if test_preds[idx].item()==labels[idx].item() else "red"))

if __name__ == '__main__':
    main()
#classfiactionohttps://gitee.cm/DK-Jun/csdn/tree/main/Pytorch%20Image%20Classification/Pytorch%20CIFAR-10%20