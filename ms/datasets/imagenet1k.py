#!/usr/bin/python
"""
Adapted from : https://github.com/tribhuvanesh/knockoffnets
"""
import os.path as osp

import ms.config as cfg
import numpy as np
from ms.utils.utils import load_dir
from torchvision.datasets import ImageFolder


class ImageNet1k(ImageFolder):
    test_frac = 0.0  # 0.2

    def __init__(self, train=True, transform=None, target_transform=None):
        root = load_dir('ILSVRC2012', 'http://image-net.org/download-images', cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC)
        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'training_imgs'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs


class ImageNet1kval(ImageFolder):
    test_frac = 0.0

    def __init__(self, train=True, transform=None, target_transform=None):
        root = load_dir('ILSVRC2012', 'http://image-net.org/download-images', cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC)

        super().__init__(root=osp.join(root, 'val'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs


class ImageNet1k32(ImageFolder):
    test_frac = 0.0  #

    def __init__(self, train=True, transform=None, target_transform=None):
        root = load_dir('ILSVRC2012', 'http://image-net.org/download-images', cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC)

        path = osp.join(root, 'val_32')

        super().__init__(root=path, transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples from path{}'.format(
            self.__class__.__name__, 'train' if train else 'test', len(self.samples), path))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs


class ImageNet1k64(ImageFolder):
    test_frac = 0.0  #

    def __init__(self, train=True, transform=None, target_transform=None):
        root = load_dir('ILSVRC2012', 'http://image-net.org/download-images', cfg.DATASET_ROOT, cfg.DATASET_ROOT_CC)

        super().__init__(root=osp.join(root, 'val_64'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs
