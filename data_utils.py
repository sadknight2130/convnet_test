from torchvision import datasets, transforms
import torch
import os


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
