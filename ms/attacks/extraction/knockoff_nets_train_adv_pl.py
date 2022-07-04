#!/usr/bin/python
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime
from math import ceil

import ms.config as cfg
import numpy as np
import torch
import wandb
from PIL import Image
from ms import datasets
from ms.utils.model_pl import LitModuleAdversary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str,
                        help='Directory containing the output model and/ortransferset.pickle')
    parser.add_argument('model_arch', metavar='model_arch', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Name of transferset dataset')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')  # this was 64
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr_step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--suffix', type=str, help='Add a suffix to modelname', default='')
    parser.add_argument('--epsilon', metavar='EPS', type=str, default=None,
                        help='Value of epsilons that will be used for extraction.')
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--scheduler_choice', type=str, help='Scheduler', default='step',
                        choices=('step', 'onecycle', 'step_adam'))
    parser.add_argument('--transferset_dir', type=str, help='Directory containing transferset.pickle')

    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    # ----------- Set up transferset
    if 'fgsm' in model_dir:
        attack = f'-fgsm-{params["epsilon"]}'
    elif 'pgd' in model_dir:
        attack = f'-pgd-{params["epsilon"]}'
    else:
        attack = ''
    transferset_file_name = f'transferset{attack}.pickle'
    transferset_dir_name = model_dir
    if params['transferset_dir']:
        transferset_dir_name = params['transferset_dir']
        transferset_path = osp.join(transferset_dir_name, transferset_file_name)
        if not osp.exists(transferset_path):
            transferset_dir_name = model_dir

    transferset_path = osp.join(transferset_dir_name, transferset_file_name)

    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i, dtype=torch.long)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples

    # ----------- Set up testset
    valid_datasets = datasets.__dict__.keys()
    dataset_name = params['testdataset']
    # we need a dataset params in adversary model creation
    params['dataset'] = params['testdataset']
    params['arch'] = params['model_arch']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    # Transferset  is picked from ImageNet so wee need to resize it using Imagenet transformations
    queryset = params['queryset']
    if queryset not in valid_datasets:
        raise ValueError('Queryset not found. Valid arguments = {}'.format(valid_datasets))
    querysetfamily = datasets.dataset_to_modelfamily[queryset]
    transferset_transform = datasets.modelfamily_to_transforms[querysetfamily]['test']

    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    BATCH_SIZE = params['batch_size']  # 256 if AVAIL_GPUS else 64
    print(BATCH_SIZE)
    NUM_WORKERS = int(os.cpu_count() / 2)

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transferset_transform)

        print()
        print('=> Training at budget = {}'.format(len(transferset)))

        print(params)

        checkpoint_suffix = '.{}'.format(b)

        train_loader = DataLoader(transferset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=False)
        if params['testdataset'] is not None:
            test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS,
                                     pin_memory=False)
        else:
            test_loader = None
        steps_per_epoch = ceil(len(transferset) / BATCH_SIZE)
        model = LitModuleAdversary(learning_rate=params['lr'], steps_per_epoch=steps_per_epoch, params=params,
                                   num_classes=num_classes)
        suffix = '-knockoff-pretrained' if params['pretrained'] else '-knockoff'
        if params['suffix']:
            suffix += '-' + params['suffix']

        model_name = f"{params['testdataset']}-{params['model_arch']}-{params['epochs']}{suffix}{attack}{checkpoint_suffix}"

        tags = ['knockoff', dataset_name]
        if attack:
            if 'pgd' in attack:
                tags.extend(['robust', 'pgd'])
            elif 'fgsm' in attack:
                tags.extend(['robust', 'fgsm'])

        group = params['dataset'] if cfg.AVAIL_GPUS == 1 else model_name
        run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY, name=model_name,
                         job_type='extraction_attack', tags=tags,
                         # id=model_name,
                         reinit=True, group=group)  # , id=model_name)
        ###  TODO this should not be saving the max ?

        attack_ckpt = attack + '-' if attack else ""

        checkpoint_callback = ModelCheckpoint(filename=f'model_knockoff{attack_ckpt}{b}', monitor="val_acc", mode="max",
                                              dirpath=model_dir)
        early_stopping_callbak = EarlyStopping(monitor='val_acc', patience=10, mode='max')

        trainer = Trainer(
            default_root_dir=model_dir,
            progress_bar_refresh_rate=cfg.REFRESH_RATE,
            max_epochs=params['epochs'],
            gpus=cfg.AVAIL_GPUS,  # num_nodes=cfg.NUM_NODES,
            accelerator=cfg.ACCELERATOR,
            # auto_select_gpus=True,
            logger=WandbLogger(project=cfg.WB_PROJECT, save_dir="lightning_logs/", name=model_name,
                               # id=model_name,
                               log_model=False),
            callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback
                       ],
        )

        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)
        trainer.test(model, test_dataloaders=test_loader)

        # Store arguments
        params['created_on'] = str(datetime.now())
        params_out_path = osp.join(model_dir, f'params_train-{attack_ckpt}{b}.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
