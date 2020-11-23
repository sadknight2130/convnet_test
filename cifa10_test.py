
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from model.convnet import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.001
num_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

val_data = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_image, val_label = next(iter(val_loader))

model = LeNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(num_epochs):
    running_loss = 0.
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 200 == 0:
            with torch.no_grad():
                val_image = val_image.to(device)
                val_label = val_label.to(device)
                output = model(val_image)
                pred = output.argmax(dim=1)  # batch_size*1
                acc = pred.eq(val_label).sum().item() / batch_size
                print("Epoch: [{}/{}], Step: [{}/{}], Loss: {}, Val_acc: {}".format(
                    epoch, num_epochs, idx, len(train_loader), running_loss/200, acc))
                running_loss = 0

print("finished training")






