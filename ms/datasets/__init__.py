from torchvision import transforms

import numbers
import numpy as np
from PIL import ImageFilter

from torchvision.datasets import ImageFolder
from ms.datasets.caltech256 import Caltech256
from ms.datasets.cifarlike import CIFAR10, CIFAR100, SVHN, TinyImagesSubset
from ms.datasets.cubs200 import CUBS200
from ms.datasets.imagenet1k import ImageNet1k, ImageNet1kval, ImageNet1k32, ImageNet1k64
from ms.datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST


# Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/11
class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception("`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception("radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception("`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


normalize_cifar10 = transforms.Normalize(
    mean=(0.491, 0.482, 0.447),
    std=(0.247, 0.244, 0.262),
)
# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_modelfamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'EMNIST': 'mnist',
    'EMNISTLetters': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'CIFAR10': 'cifar',
    'CIFAR100': 'cifar',
    'SVHN': 'cifar',
    'TinyImageNet200': 'cifar',
    'TinyImagesSubset': 'cifar',

    'GTSRB': 'gtsrb',

    # Imagenet
    'CUBS200': 'imagenet',
    'Caltech256': 'imagenet',
    'Indoor67': 'imagenet',
    'Diabetic5': 'imagenet',
    'ImageNet1k': 'imagenet',
    'ImageNet1kval': 'imagenet',
    'ImageNet1k32': 'imagenet32',
    'ImageNet1k64': 'imagenet64',
    'ImageFolder': 'imagenet',
}

modelfamily_to_mean_std = {
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,)
    },
    'cifar': {
        'mean': (0.491, 0.482, 0.447),  # (x / 255.0 for x in [125.3, 123.0, 113.9]),
        'std': (0.247, 0.244, 0.262)  # (x / 255.0 for x in [63.0, 62.1, 66.7]),
        # 'mean': (0.4914, 0.4822, 0.4465),
        # 'std': (0.2023, 0.1994, 0.2010),
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
}

modelfamily_to_inv_mean_std = {
    'mnist': {
        'mean': [(0.,),
                 (-0.1307,)],
        'std': [(1 / 0.3081,),
                (1.,)],
    },
    'cifar': {
        'mean': [(0., 0., 0.),
                 (-0.4914, -0.4822, -0.4465)],
        'std': [(1 / 0.2023, 1 / 0.1994, 1 / 0.2010),
                (1., 1., 1.)],
    },
    'imagenet': {
        'mean': [(0., 0., 0.),
                 (-0.485, -0.456, -0.406)],
        'std': [(1 / 0.229, 1 / 0.224, 1 / 0.225),
                (1., 1., 1.)],
    }
}

# Transforms
modelfamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            # transforms.Resize((32, 32)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_cifar10,
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
            #                      std=(0.2023, 0.1994, 0.2010)),
        ]),
        'test': transforms.Compose([
            transforms.Resize((32, 32)),  # this has been added to resize
            transforms.ToTensor(),
            normalize_cifar10,
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
            #                      std=(0.2023, 0.1994, 0.2010)),
        ])
    },

    'gtsrb': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_cifar10,
            # transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669)),
            # # mean_nums = [0.485, 0.456, 0.406]
            # # std_nums = [0.229, 0.224, 0.225]
            # transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
            #                      std=[ 0.2672, 0.2564, 0.2629])
        ]),
        'test': transforms.Compose([
            # transforms.RandomCrop(32, padding=4),  
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize_cifar10, 
            # transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
            #                      std=[0.2672, 0.2564, 0.2629])
        ])
    },

    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  
            # transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((224, 224)), 
            transforms.Resize(256),  
            transforms.CenterCrop(224),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet32': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),  
            # transforms.Resize((32, 32)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((32, 32)), 
            transforms.Resize(32),  
            transforms.CenterCrop(32),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet64': {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((64, 64)), 
            transforms.Resize(64),  
            transforms.CenterCrop(64),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    }
}

"""
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# -------------NO NORMALIZATION FOR THIS DATA -  TO BE USED ------------- #
# ----------------IN ADVERSARIAL EXAMPLES GENERATION -------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
"""

# Transforms
modelfamily_to_transforms_without_normalization = {
    'mnist': {
        'train': transforms.Compose([
            transforms.Resize(28),  
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(28),  
            transforms.ToTensor(),
        ]),
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            # transforms.Resize((32, 32)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((32, 32)),  # this has been added to resize
            transforms.ToTensor(),
        ])
    },

    'gtsrb': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            # transforms.RandomCrop(32, padding=4),  
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    },

    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  
            # transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((224, 224)), 
            transforms.Resize(256),  
            transforms.CenterCrop(224),  
            transforms.ToTensor(),
        ])
    },

    'imagenet32': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),  
            # transforms.Resize((32, 32)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((32, 32)), 
            transforms.Resize(32),  
            transforms.CenterCrop(32),  
            transforms.ToTensor(),
        ])
    },

    'imagenet64': {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            # transforms.Resize((64, 64)), 
            transforms.Resize(64),  
            transforms.CenterCrop(64),  
            transforms.ToTensor(),
        ])
    }
}
