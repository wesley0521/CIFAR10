import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torchvision.models import ResNet18_Weights


device = torch.device("cuda")
# 使用 ResNet 作為模型
weights=ResNet18_Weights.DEFAULT
ResNet = torchvision.models.resnet18(weights=weights)
ResNet.fc = torch.nn.Linear(ResNet.fc.in_features, 10)
ResNet = ResNet.to(device)
print(ResNet)


transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor()
])


train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
print(f"訓練集長度 : {len(train_data)}")
print(f"驗證集長度 : {len(test_data)}")

train_dataloader = DataLoader(train_data, batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# 設定損失函數
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device=device)
# 最佳化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(ResNet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)


# 開始訓練
epoch = 100
train_step = 0
test_step = 0
writer = SummaryWriter("ResNet")
for i in range(epoch):
    print(f"-----第{i+1}輪訓練開始-----")

    ResNet.train()
    scheduler.step()
    for train_data in train_dataloader:
        train_imgs, train_targets = train_data
        train_imgs, train_targets = train_imgs.to(device), train_targets.to(device)
        # 放入模型
        train_output = ResNet(train_imgs)
        train_loss = loss_fn(train_output, train_targets)
        # 最佳化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 100 == 0:
            print(f"Loss : {train_loss} ; 訓練次數 : {train_step}")
            writer.add_scalar("train_loss", train_loss.item(), global_step=train_step)

    total_test_loss = 0
    total_accuracy = 0
    ResNet.eval()
    with torch.no_grad():
        for data in test_dataloader:
            test_imgs, test_targets = data
            test_imgs, test_targets = test_imgs.to(device), test_targets.to(device)
            test_output = ResNet(test_imgs)
            test_loss = loss_fn(test_output, test_targets)
            total_test_loss = total_test_loss + test_loss.item()
            # 計算 accuracy
            accuracy = (test_output.argmax(1) == test_targets).sum().item()
            total_accuracy = total_accuracy + accuracy
    total_accuracy_per = total_accuracy / (len(test_data))
    print(f"驗證集的 loss : {total_test_loss}")
    print(f"驗證集的 accuracy : {total_accuracy_per}")
    writer.add_scalar("test_loss", total_test_loss, test_step)
    writer.add_scalar("test_accuracy", total_accuracy_per, test_step)
    test_step += 1
    
writer.close()
