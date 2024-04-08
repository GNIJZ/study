import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST



# 定义变分自编码器的编码器部分
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)

        return mean, logvar


# 定义变分自编码器的解码器部分
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        recon_x = torch.sigmoid(self.fc2(z))  # 由于使用MNIST等数据，使用sigmoid激活函数
        return recon_x


# 定义变分自编码器
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar


# 定义损失函数：重构损失 + KL 散度
def vae_loss(recon_x, x, mean, logvar):
    # 重构损失
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784),
    #                                     reduction='sum')
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return  kl_divergence+recon_loss+recon_loss


# 定义训练函数
def train_vae(model, train_loader, optimizer, num_epochs):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0

        for imgs, _ in train_loader:
            bs = imgs.size(0)
            imgs = imgs.to(device).view(bs, -1)
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(imgs)
            loss = vae_loss(recon_batch, imgs, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}, Average Loss: {:.4f}'.format(epoch + 1, total_loss / len(train_loader.dataset)))


# 示例用法
input_dim = 28 * 28  # MNIST数据集的输入维度
hidden_dim = 512
latent_dim = 300  # 潜在空间的维度
learning_rate = 1e-3
num_epochs = 50

model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(model)
train_vae(model.to(device), train_loader, optimizer, num_epochs)
