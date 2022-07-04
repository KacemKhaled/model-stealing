import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)


# Paths: cfg.DATASET_ROOT, cfg.DATASET_ROOT_SDD, cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC
def load_dir(folder, url='URL', *paths):
    root = ""
    for path in paths:
        root = osp.join(path, folder)
        if osp.exists(root):
            return root
    if not osp.exists(root):
        raise ValueError(f'Dataset not found at {paths}. Please download it from {url}.')

# Paths: cfg.DATASET_ROOT, cfg.DATASET_ROOT_SDD, cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC

def load_model_dir(folder, *paths):
    root = ""
    for path in paths:
        root = osp.join(path, folder)
        if osp.exists(root):
            return root
    if not osp.exists(root):
        raise ValueError(f'Dataset not found at {paths}.')


class AdvSetImagePaths(ImageFolder):
    """adv_train_set Dataset, for when images are stored as *paths*"""
    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class AdvSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if img.shape == (28, 28):  # CIFAR 10 : (3, 32, 32):
            # for MNIST we should do this :
            img = Image.fromarray(img)
        else:
            # skipping img tranformation since it is already scaled between [0. ,1.]
            self.transform = None

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Image cannot be transformed')

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_adv_data_set(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    if budget is None or budget > len(samples):
        budget = len(samples)
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return AdvSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return AdvSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def my_collate(batch):
    """Define collate_fn myself because to use for cancatenation"""
    # item: a tuple of (img, label)
    data = [torch.Tensor(item[0]) for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]
