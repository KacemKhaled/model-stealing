#!/usr/bin/python

import argparse
import json
import os
import os.path as osp
import pickle
import random
from datetime import datetime
from math import ceil

import ms.config as cfg
import ms.utils.utils as ms_utils
import numpy as np
import torch
import wandb
from ms import datasets
from ms.utils.model_pl import LitModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset, random_split


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing victim model')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to transfer set', required=True)
    parser.add_argument('model_arch', metavar='model_arch', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--ratio', metavar='B', type=float,
                        help='Comma separated values of ratio of benign samples and adversarial samples. \
                        Robust will be trained for a concatenation of both.')
    parser.add_argument('--attack', metavar='TYPE', type=str, default='fgsm-0.03',
                        help='Name of attack used to generate adversarial examples')
    parser.add_argument('--epsilons', metavar='EPS', type=str,
                        help='Comma separated values of epsilons that will be used for adversarial training.')

    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
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
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--scheduler_choice', type=str, help='Scheduler', default='step',
                        choices=('step', 'onecycle', 'step_adam'))
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']
    out_path = params['out_dir']
    ms_utils.create_dir(out_path)

    # ----------- Set up adv_train_set
    attacks = params['attack'].split(',')
    epsilons = [float(b) for b in params['epsilons'].split(',')]
    for attack in attacks:
        for eps in epsilons:

            adv_set_file = f"{attack}-{eps}"  # params['adv_set_file']
            print(f"INFO: using {adv_set_file}")
            adv_train_set_path = osp.join(model_dir, f'adv_data/{params["testdataset"]}-train-{adv_set_file}.pickle')
            if not osp.exists(adv_train_set_path):
                print(f'Adv train samples not found at {adv_train_set_path}.')
                continue
            with open(adv_train_set_path, 'rb') as rf:
                adv_train_set_samples = pickle.load(rf)
            adv_test_set_path = osp.join(model_dir, f'adv_data/{params["testdataset"]}-test-{adv_set_file}.pickle')
            if not osp.exists(adv_test_set_path):
                print(f'Adv test samples not found at {adv_test_set_path}.')
                continue
            with open(adv_test_set_path, 'rb') as rf:
                adv_test_set_samples = pickle.load(rf)
            num_classes = max(x[1] for x in adv_train_set_samples) + 1
            print(
                '=> found adversarial set with {} samples, {} classes'.format(len(adv_train_set_samples), num_classes))

            # ----------- Set up testset
            valid_datasets = datasets.__dict__.keys()
            dataset_name = params['testdataset']
            # we need a dataset params in adversary model creation
            params['dataset'] = params['testdataset']
            params['arch'] = params['model_arch']
            if dataset_name not in valid_datasets:
                raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
            modelfamily = datasets.dataset_to_modelfamily[dataset_name]
            test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
            train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']

            dataset = datasets.__dict__[dataset_name]
            trainset = dataset(train=True, transform=train_transform)
            testset = dataset(train=False, transform=test_transform)

            # ----------- Set up model
            model_name = params['model_arch']
            pretrained = params['pretrained']

            AVAIL_GPUS = min(1, torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
            BATCH_SIZE = params['batch_size']  # 256 if AVAIL_GPUS else 64
            print(BATCH_SIZE)
            NUM_WORKERS = int(os.cpu_count() / 2)

            # ----------- Train
            ratio_adv = params['ratio']

            np.random.seed(cfg.DEFAULT_SEED)
            torch.manual_seed(cfg.DEFAULT_SEED)
            torch.cuda.manual_seed(cfg.DEFAULT_SEED)
            generator = torch.Generator().manual_seed(cfg.DEFAULT_SEED)

            # TODO make this as mix for adv and benign samples
            adv_train_set = ms_utils.samples_to_adv_data_set(adv_train_set_samples, transform=test_transform)
            adv_test_set = ms_utils.samples_to_adv_data_set(adv_test_set_samples, transform=test_transform)
            if 0 < ratio_adv < 1:
                # here we mix from both the original data and adversarial data
                print(
                    f'INFO: Using a ratio of {ratio_adv:.2%} of adversarial samples and {1 - ratio_adv:.2%} of normal samples')
                indices = set(range(len(adv_train_set)))

                indices_adv = set(random.sample(indices, int(ratio_adv * len(adv_train_set))))
                indices_benign = set(range(len(trainset))) - indices_adv

                trainset = Subset(trainset, list(indices_benign))
                adv_train_set = Subset(adv_train_set, list(indices_adv))
                print(f'Benign samples {len(trainset)}, training set size = {len(adv_train_set)}')
                adv_train_data = trainset + adv_train_set
            elif ratio_adv == 0:
                print(
                    f'INFO: Using a ratio of {ratio_adv:.2%} of adversarial samples and {1 - ratio_adv:.2%} of normal samples')
                adv_train_data = trainset
            elif ratio_adv > 1:
                print(f'INFO: Using all of adversarial samples and the normal samples together')
                adv_train_data = trainset + adv_train_set
            else:
                print(f'INFO: Using a ratio of 100% of adversarial samples and 0% of normal samples')
                adv_train_data = adv_train_set

            print()
            print(f'=> Training at ratio {ratio_adv}, training set size = {len(adv_train_data)}')
            print(params)
            checkpoint_suffix = f'.r{ratio_adv}'
            validation_split = .2

            train_size = int((1 - validation_split) * len(adv_train_data))
            val_size = len(adv_train_data) - train_size
            train_set, val_set = random_split(adv_train_data, [train_size, val_size], generator=generator)
            print(f"Train set : {train_size}\nValidation set: {val_size}")
            adv_train_loader = DataLoader(train_set, collate_fn=ms_utils.my_collate,
                                          batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                          pin_memory=False)
            validation_loader = DataLoader(val_set, collate_fn=ms_utils.my_collate,
                                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                           pin_memory=False)
            adv_test_loader = DataLoader(adv_test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                         pin_memory=False)
            if params['testdataset'] is not None:
                test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS,
                                         pin_memory=False)
            else:
                test_loader = None
            steps_per_epoch = ceil(len(adv_train_data) / BATCH_SIZE)
            model = LitModule(learning_rate=params['lr'], steps_per_epoch=steps_per_epoch, params=params,
                              num_classes=num_classes)
            suffix = '-robust-pretrained' if params['pretrained'] else '-robust'

            model_name = f"{params['testdataset']}-{params['model_arch']}-{params['epochs']}{suffix}-{adv_set_file}{checkpoint_suffix}"
            run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY, name=model_name,
                             job_type='adversarial_training',
                             reinit=True, group=params['testdataset'])
            wandb.log({'eps': eps})
            checkpoint_callback = ModelCheckpoint(filename=f'model_robust-{adv_set_file}', monitor="val_acc",
                                                  mode="max",
                                                  dirpath=out_path)
            early_stopping_callbak = EarlyStopping(monitor='val_acc', patience=10, mode='max')
            trainer = Trainer(
                default_root_dir=out_path,
                progress_bar_refresh_rate=cfg.REFRESH_RATE,
                max_epochs=params['epochs'],  # 30
                gpus=cfg.AVAIL_GPUS,  # num_nodes=cfg.NUM_NODES,
                accelerator=cfg.ACCELERATOR,
                logger=WandbLogger(project=cfg.WB_PROJECT, save_dir="lightning_logs/", name=model_name,
                                   id=wandb.util.generate_id(),
                                   log_model=False),
                callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback,
                           ],
            )

            # load checkpoint

            trainer.fit(model, train_dataloader=adv_train_loader, val_dataloaders=validation_loader)
            test_loaders = [test_loader, adv_test_loader]
            trainer.test(model, test_dataloaders=test_loaders)

            # Store arguments
            params['created_on'] = str(datetime.now())
            params_out_path = osp.join(out_path, f'params_train-{adv_set_file}.json')
            with open(params_out_path, 'w') as jf:
                json.dump(params, jf, indent=True)
            wandb.finish()


if __name__ == '__main__':
    main()
