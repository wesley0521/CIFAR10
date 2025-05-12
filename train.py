import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 建立模型
CNN_Model = CNN_Model()
CNN_Model = CNN_Model.to(device=device)
# 設定損失函數
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device=device)
# 最佳化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(CNN_Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)


# 開始訓練
epoch = 100
train_step = 0
test_step = 0
writer = SummaryWriter("CNN")
for i in range(epoch):
    print(f"-----第{i+1}輪訓練開始-----")

    CNN_Model.train()
    scheduler.step()
    for train_data in train_dataloader:
        train_imgs, train_targets = train_data
        train_imgs, train_targets = train_imgs.to(device), train_targets.to(device)
        # 放入模型
        train_output = CNN_Model(train_imgs)
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
    CNN_Model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            test_imgs, test_targets = data
            test_imgs, test_targets = test_imgs.to(device), test_targets.to(device)
            test_output = CNN_Model(test_imgs)
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
