import torch
from torchvision.datasets import ImageFolder
from utils import get_transforms, get_albumentations_transforms, AlbumentationsImageFolder
from torch.utils.data import DataLoader
import os


def prepare_data(data_dir, batch_size=32):

    train_dataset = AlbumentationsImageFolder(root = os.path.join(data_dir,"train/images"), transform=get_albumentations_transforms(train=True))
    test_dataset = AlbumentationsImageFolder(root = os.path.join(data_dir,"validation/images"),transform=get_albumentations_transforms(train=False))

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names

