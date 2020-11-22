from torchvision import datasets, transforms
import torch
import os
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir, input_size, batch_size, mode_dict):
    data_transforms = {
        "train": transforms.Compose(
            [transforms.RandomResizedCrop(input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in mode_dict}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size, shuffle=True, num_workers=0) for x in
                        mode_dict}
    return dataloaders_dict


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)
