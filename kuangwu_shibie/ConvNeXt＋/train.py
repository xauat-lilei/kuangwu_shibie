import os
import argparse

import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")



    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        val_loss_list.append( val_loss)
        val_acc_list.append(val_acc )


        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['val_accurate'] = val_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['val_loss'] = val_loss_list
    print('Finished Training')
    import huatu
    huatu.plot_history(args.epochs, Acc, Loss)
    # 保存数据
    import random
    import pandas as pd
    from datetime import datetime
    # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df1 = pd.DataFrame(columns=[ 'train_acc_list', 'val_acc_list'])  # 列名
    df1.to_csv("E:\\李雷\\矿石训练最终程序\\矿石图像训练\\ConvNeXt＋\\acc.csv", index=False)  # 路径可以根据需要更改

    df2 = pd.DataFrame(columns=['train_loss_list', 'val_loss_list'])  # 列名
    df2.to_csv("E:\\李雷\\矿石训练最终程序\\矿石图像训练\\ConvNeXt＋\\loss.csv", index=False)  # 路径可以根据需要更改
    # 将数据保存在一维列表
    # 保存准确率数据
    list1 = [ train_acc_list, val_acc_list]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data1 = pd.DataFrame([list1])
    data1.to_csv('E:\\李雷\\矿石训练最终程序\\矿石图像训练\\ConvNeXt＋\\acc.csv', mode='a', header=False,
                 index=False)  # mode设为a,就可以向csv文件追加数据了

    # 保存LOSS数据
    list2 = [train_loss_list, val_loss_list]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data2 = pd.DataFrame([list2])
    data2.to_csv('E:\\李雷\\矿石训练最终程序\\矿石图像训练\\ConvNeXt＋\\loss.csv', mode='a', header=False,
                 index=False)  # mode设为a,就可以向csv文件追加数据了

    ###################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=26)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="E:\\李雷\\矿石训练最终程序\\矿石图像训练\\dtat\\kuangshi_photo")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
