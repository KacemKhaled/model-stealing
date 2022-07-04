#
import argparse
import json
import os
import os.path as osp
from datetime import datetime

import ms.config as cfg
import ms.utils.utils as ms_utils
import torch
import torchvision.models as models
import wandb
from ms.utils.model_pl import genericDataModule, LitModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def argparser():
    parser = argparse.ArgumentParser(description='Train models, pytorch-lightning parallel GPU')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    print(model_names)
    valid_mnist_models = ['cnn', 'lenet']

    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('datapath', metavar='DIR', default=cfg.DATASET_ROOT,
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names + valid_mnist_models,
                        help='model architecture: ' +
                             ' | '.join(model_names + valid_mnist_models) +
                             ' (default: resnet18)')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate (default: 0.05)', dest='lr')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--ev', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model', default=False)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-s', '--scheduler_choice', type=str, help='Scheduler', default='step',
                        choices=('step', 'onecycle', 'step_adam'))
    return parser.parse_args()


seed_everything(7)


def main():
    print("Starting...")

    args = argparser()
    params = vars(args)
    PATH_DATASETS = os.path.join(params['datapath'], 'train')

    print(torch.cuda.get_device_name(0))
    BATCH_SIZE = params['batch_size']
    print(BATCH_SIZE)
    NUM_WORKERS = int(os.cpu_count() / 2)
    dm = genericDataModule(params['dataset'], 0.2, BATCH_SIZE, NUM_WORKERS)
    MODEL_DIR = params['out_path']
    if not osp.exists(MODEL_DIR):
        ms_utils.create_dir(MODEL_DIR)
    model = LitModule(learning_rate=params['lr'], steps_per_epoch=dm.steps_per_epoch, params=params,
                      num_classes=dm.num_classes)
    suffix = '--pretrained' if params['pretrained'] else ''
    model_name = f"{params['dataset']}-{params['arch']}-{params['epochs']}{suffix}"
    group = params['dataset'] if cfg.AVAIL_GPUS == 1 else model_name

    run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY, name=model_name,
                     job_type='training',
                     group=group,
                     reinit=False)
    checkpoint_callback = ModelCheckpoint(filename='model_best', monitor="val_acc", mode="max",
                                          dirpath=MODEL_DIR)
    early_stopping_callbak = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    trainer = Trainer(
        default_root_dir=MODEL_DIR,
        progress_bar_refresh_rate=cfg.REFRESH_RATE,
        max_epochs=params['epochs'],  # 30
        gpus=cfg.AVAIL_GPUS,  # num_nodes=cfg.NUM_NODES,
        accelerator=cfg.ACCELERATOR,
        logger=WandbLogger(project=cfg.WB_PROJECT, save_dir="lightning_logs/", name=model_name,
                           log_model=False),
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback,
                   ],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(MODEL_DIR, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    wandb.finish()


if __name__ == '__main__':
    main()
