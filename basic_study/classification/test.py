import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms
# from torchsummary import summary


print(torch.__version__)
# 创建神经网络（有注释）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 500, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(500, 50, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_layer = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x)
        return output

# 超参数
EPOCH = 50
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD = False  # 若已经下载mnist数据集则设为False

# 使用cuda加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载mnist数据
train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD
)

# DataLoader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0 if device.type == 'cuda' else 2  # 如果使用 GPU 加速，将 num_workers 设置为 0
)
print(device)
# 新建网络
cnn = CNN().to(device)

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# 损失函数
loss_func = nn.CrossEntropyLoss()

# 测试集
test_data = datasets.MNIST(
    root='./data',
    train=False
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_x = test_x.to(device)

test_y = test_data.targets[:2000]
test_y = test_y.to(device)

# 训练神经网络
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔50步输出一次信息
        if step % 50 == 0:
            test_output = cnn(test_x)
            predict_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (predict_y == test_y).sum().item() / test_y.size(0)
            print('Epoch', epoch, '|', 'Step', step, '|', 'Loss', loss.item(), '|', 'Test Accuracy', accuracy)

# 预测
test_output = cnn(test_x[:100])
predict_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
real_y = test_y[:100].cpu().numpy()

print(predict_y)
print(real_y)

# 打印预测和实际结果
for i in range(10):
    print('Predict', predict_y[i])
    print('Real', real_y[i])
