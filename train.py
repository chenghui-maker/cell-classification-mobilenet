from __future__ import print_function
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import mobilenet_v3_small






class myDataset(Dataset):
    def __init__(self, lines, input_transforms):
        self.lines = lines
        self.input_transforms = input_transforms

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.split(";")
        img_id = line[1]
        root_dir = "./data"
        img_path = os.path.join(root_dir, img_id)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'))
        img = self.input_transforms(img)
        label = int(line[0])
        return img, label

    def __len__(self):
        return len(self.lines)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomVerticalFlip(),  # 垂直翻转
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.1),    # 调节亮度
            transforms.ColorJitter(contrast=0.3),      # 调节对比度
            transforms.ColorJitter(saturation=0.1),    # 调节饱和度
            transforms.GaussianBlur(kernel_size=5)]),  # 模糊处理 默认sigma取值（0.1, 2.0）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def val_transform():
    return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


Batch_size = 64
Init_epoch = 0
Adam_epoch = 5
Epoch = 10

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===> device: {}".format(device))

# 加载数据
print('===> Loading datasets')
rootDataFlie = "./datafile"
train_path = os.path.join(rootDataFlie, "train_cleanest_smaple.txt")
val_path = os.path.join(rootDataFlie, "val_cleanest.txt")
with open(train_path, "r") as f:
    train_lines = f.read().splitlines()
with open(val_path, "r") as f:
    val_lines = f.read().splitlines()


train_transform = train_transform()
train_set = myDataset(train_lines, input_transforms=train_transform)
val_transform = val_transform()
val_set = myDataset(val_lines, input_transforms=val_transform)
training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=Batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=Batch_size, shuffle=True)
print("训练集数据量: {}".format(len(train_set)))
print("验证集数据量: {}".format(len(val_set)))

# 搭建模型
print('===> Building model')
model = mobilenet_v3_small(pretrained=True)
in_features = model.classifier[0].out_features
model.classifier[3] = nn.Linear(in_features, 2)  # 2类
print(model)
model = model.to(device)



# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)
#添加 tensorboard
SummaryWriter_path = "./tensorboard_logs/mobilenet_v3_small_cleanest"
if not os.path.exists(SummaryWriter_path):
    os.makedirs(SummaryWriter_path)
writer = SummaryWriter(log_dir=SummaryWriter_path)

total_train_step = 0  #记录训练次数
total_test_step = 0   #记录测试次数


save_model_path = "./model_weights/mobilenet_v3_small_cleanest"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# 优化器为Adam
if True:
    lr = 1e-4
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    # 学习率
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    for i in range(Init_epoch, Adam_epoch):
        print("---------第{}轮训练开始----------".format(i + 1))
        model.train()
        for data in training_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 50 == 0:
                print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=total_train_step)

        # 验证
        total_test_loss = 0
        total_acc = 0
        model.eval()
        with torch.no_grad():
            for data in testing_data_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                acc = torch.eq(outputs.argmax(1), targets).sum()

                total_acc = total_acc + acc.item()
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()

        print("整体测试数据的Loss :{}".format(round(total_test_loss, 3)))
        print("整体测试数据的acc :{}".format(round(total_acc / len(val_set), 5)))
        writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step)
        writer.add_scalar(tag="test_acc", scalar_value=total_acc / len(val_set), global_step=total_test_step)
        total_test_step = total_test_step + 1

        # 保存模型
        torch.save(model.state_dict(),
                   save_model_path + "/Class_mobilenetv3_small_model_{}_val_loss_{}_acc_{}.pth".format(
                   (i + 1), round(total_test_loss, 3), round(total_acc / len(val_set), 5)))

        lr = get_lr(optimizer)
        print("第{}轮优化器是: {} ".format((i + 1), str(optimizer).split("(")[0]))
        print("第{}轮学习率是: {} ".format((i + 1), lr))
        lr_scheduler.step()



# 优化器为SGD
if True:
    lr = 1e-4
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    # 学习率
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    for i in range(Adam_epoch, Epoch):
        print("---------第{}轮训练开始----------".format(i + 1))
        model.train()
        for data in training_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 50 == 0:
                print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=total_train_step)

        # 验证
        total_test_loss = 0
        total_acc = 0
        model.eval()
        with torch.no_grad():
            for data in testing_data_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                acc = torch.eq(outputs.argmax(1), targets).sum()

                total_acc = total_acc + acc.item()
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()

        print("整体测试数据的Loss :{}".format(round(total_test_loss, 3)))
        print("整体测试数据的acc :{}".format(round(total_acc / len(val_set), 5)))
        writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step)
        writer.add_scalar(tag="test_acc", scalar_value=total_acc / len(val_set), global_step=total_test_step)
        total_test_step = total_test_step + 1

        # 保存模型
        torch.save(model.state_dict(),
                   save_model_path + "/Class_mobilenetv3_small_model_{}_val_loss_{}_acc_{}.pth".format(
                       (i + 1), round(total_test_loss, 3), round(total_acc / len(val_set), 5)))


        lr = get_lr(optimizer)
        print("第{}轮优化器是: {} ".format((i + 1), str(optimizer).split("(")[0]))
        print("第{}轮学习率是: {} ".format((i + 1), lr))
        lr_scheduler.step()

