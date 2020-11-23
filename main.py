import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from model.convnet import Net


from model.train_eval import train, test


if __name__ == "__main__":
    train_data = datasets.MNIST("./mnist_data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_data = datasets.MNIST("./mnist_data", train=False, transform=transforms.ToTensor())
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

    torch.save(model.state_dict(), "mnist_cnn.pt")