import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils.convnet import Net
import os


from utils.train_eval import train, test


if __name__ == "__main__":
    dataroot = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = dataroot + "/data/mnist_data"
    train_data = datasets.MNIST(image_path, train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_data = datasets.MNIST(image_path, train=False, transform=transforms.ToTensor())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    lr = 0.01
    momentum = 0.5
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    print(type(criterion))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    num_epochs = 2

    for epoch in range(num_epochs):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader, criterion)

