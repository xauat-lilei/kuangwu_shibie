import time

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
epochs=30
best_acc = 0
save_path = './googleNet.pth'
train_steps = len(train_loader)
train_acc_list, val_accurate_list = [], []
train_loss_list, val_loss_list = [], []
lr_list = []
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
        probs, pred_y = outputs.data.max(dim=1)  # 得到概率
        train_acc += (pred_y == labels).sum() / labels.size(0)

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * (50 - int(rate * 50))
        print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(epoch + 1, epochs, int(rate * 100), a, b), end='')

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc * 100 / len(train_loader)
    #     print('train_loss:{:.3f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)


    #validate
    net.eval()
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for step, data in enumerate(val_bar):
            val_images, val_labels = data
            val_images= val_images.to(device)
            val_labels= val_labels.to(device)
            outputs = net(val_images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.data
            probs, pred_y = outputs.data.max(dim=1)  # 得到概率
            val_accurate += (pred_y == val_labels).sum() / val_labels.size(0)
            rate = (step + 1) / len(validate_loader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i + 1, epoch, int(rate * 100), a, b), end='')
    val_loss = val_loss / len(validate_loader)
    val_accurate = val_accurate * 100 / len(validate_loader)
    val_loss_list.append(val_loss)
    val_accurate_list.append(val_accurate)
    end = time.time()
    print(
        '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
            epoch + 1, epochs, train_loss, train_acc, val_loss, val_accurate, lr), end='')
    time_ = int(end - start)
    time_ = int(end - start)
    h = time_ / 3600
    m = time_ % 3600 / 60
    s = time_ % 60
    time_str = "\tTime %02d:%02d" % (m, s)
    # 打印所用时间
    print(time_str)
    # 如果取得更好的准确率，就保存模型
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

Acc = {}
Loss = {}
Acc['train_acc'] = train_acc_list
Acc['val_accurate'] = val_accurate_list
Loss['train_loss'] = train_loss_list
Loss['val_loss'] = val_loss_list
