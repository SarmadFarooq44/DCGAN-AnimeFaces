import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("splcher/animefacedataset")

print("Path to dataset files:", path)

#path = "C:/Users/---/.cache/kagglehub/datasets/splcher/animefacedataset/versions/3"

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
latent_dim = 100          # Size of the latent noise vector
image_channels = 3        # RGB channels
hidden_dim = 64           # Base number of filters
image_size = 64           # Output image size
batch_size = 64           # Batch size for training
lr = 0.0002               # Learning rate
beta1 = 0.5               # Adam optimizer beta1
num_epochs = 50           # Number of training epochs

# 1. Data Preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load Anime Face Dataset (replace with your dataset path)
dataset = ImageFolder(root=path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # hidden_dim*8 x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # hidden_dim*4 x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # hidden_dim*2 x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # hidden_dim x 32 x 32
            nn.ConvTranspose2d(hidden_dim, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range: [-1, 1]
            # Output: image_channels x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

# 3. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden_dim x 32 x 32
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden_dim*2 x 16 x 16
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden_dim*4 x 8 x 8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden_dim*8 x 4 x 4
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability
            # Output: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Initialize networks
netG = Generator().to(device)
netD = Discriminator().to(device)

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# 4. Loss and Optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Labels for real and fake images
real_label = 1.
fake_label = 0.

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Train Discriminator
        netD.zero_grad()
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
        output = netD(real_images)
        lossD_real = criterion(output, labels)
        lossD_real.backward()

        # Generate fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake_images.detach())
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        labels.fill_(real_label)
        output = netD(fake_images)
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()

        # Print training progress
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss D: {lossD_real.item() + lossD_fake.item():.4f}, Loss G: {lossG.item():.4f}")

    # Save generated images every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            noise = torch.randn(64, latent_dim, 1, 1, device=device)
            fake_images = netG(noise).cpu()
            grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(f"generated_images_epoch_{epoch}.png")
            plt.close()

# Save the trained models
torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')

print("Training completed!")