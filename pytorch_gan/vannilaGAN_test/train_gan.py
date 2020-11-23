from __future__ import division
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from utils.data_utils import load_mnist
from pytorch_gan.vannilaGAN_test.model_gan import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=32
data_path = "../../data/mnist_data"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5,
                        std=0.5)
])

dataloader = load_mnist(data_path, transform, batch_size)

image_size = 784
hidden_size = 256
latent_size = 64
# discriminator
D = get_d(image_size, hidden_size)
# Generator
G = get_g(latent_size, hidden_size, image_size)

D = D.to(device)
G = G.to(device)

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


total_step = len(dataloader)
num_epochs = 50
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.reshape(batch_size, image_size).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        # 开始生成fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        # 开始优化discriminator
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 开始优化generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 1000 == 0:
            print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}"
                  .format(epoch, num_epochs, i, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(),
                          fake_score.mean().item()))


z = torch.randn(1, latent_size).to(device)
fake_images = G(z).view(28, 28).data.cpu().numpy()
plt.imshow(fake_images)