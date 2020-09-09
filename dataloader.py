from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import glob


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.dataset = []
        self.preprocess()

        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        self.dataset = glob.glob(f"{self.image_dir}/*")

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index, mode = "train", ):
        """Return one image."""
        dataset = self.dataset
        filename = dataset[index]
        image = Image.open(filename)
        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(batch_size, image_dir, crop_size=178,
               image_size=128,
               dataset='CelebA', mode='train', num_workers=16):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader