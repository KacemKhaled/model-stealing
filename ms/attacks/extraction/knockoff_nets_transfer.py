#!/usr/bin/python

import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import ms.config as cfg
import ms.utils.utils as ms_utils
import numpy as np
import torch
from ms import datasets
from ms.utils.test import get_model_checkpoint_path, load_best_model, load_model
from tqdm import tqdm


class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def get_transferset(self, budget):
        start_B = 0
        end_B = budget
        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, budget - len(self.transferset)))
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.queryset)))

                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                y_t = self.blackbox(x_t)  # .cpu()
                # print(y_t[:3])
                # print(y_t[:3].cpu())

                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute
                    # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself
                    # But, we need to store the non-transformed version
                    img_t = [self.queryset.data[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(x_t.size(0)):
                    img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    self.transferset.append((img_t_i, y_t[i].cpu().squeeze()))

                pbar.update(x_t.size(0))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.ckpt" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    parser.add_argument('-v', '--victim_select', type=int, help='Which victim, last: -1 or best: -2', default=0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    transferset_out_path = params['victim_model_dir']
    ms_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Available device: {device}")
    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params[
        'modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=transform)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    # select all checkpoints with select=-2

    if 'pgd' in blackbox_dir or 'fgsm' in blackbox_dir:
        checkpoint_files = [[f"{file}", f"{blackbox_dir}/{file}"] for file in os.listdir(blackbox_dir)
                            if file.endswith('-0.05.ckpt') or file.endswith('-0.03.ckpt')]
        for file, path in checkpoint_files:
            blackbox = load_model(path, with_norm_layer_on_forward_only=False)

            print(f'Using the blackbox victim model: {path}')

            # ----------- Initialize adversary
            batch_size = params['batch_size']
            nworkers = int(os.cpu_count() / 2)  # params['nworkers']

            suffix = file[file.find('-'):-5]
            # transfer_out_path = osp.join(out_path, f'transferset{suffix}.pickle')
            transfer_out_path = osp.join(transferset_out_path, f'transferset{suffix}.pickle')
            print(transfer_out_path)

            adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)

            print('=> constructing transfer set...')
            transferset = adversary.get_transferset(params['budget'])
            with open(transfer_out_path, 'wb') as wf:
                pickle.dump(transferset, wf)
            print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

            # Store arguments
            params['created_on'] = str(datetime.now())
            # params_out_path = osp.join(out_path, f'params_transfer{suffix}.json')
            params_out_path = osp.join(transferset_out_path, f'params_transfer{suffix}.json')
            with open(params_out_path, 'w') as jf:
                json.dump(params, jf, indent=True)
    else:
        selected_checkpoints = get_model_checkpoint_path(blackbox_dir, select=params['victim_select'])

        blackbox, test_acc, _, _, checkpoint_path = load_best_model(selected_checkpoints, blackbox_dir,
                                                                    with_norm_layer_on_forward_only=False)

        print(f'Using the blackbox victim model: {checkpoint_path}')

        # ----------- Initialize adversary
        batch_size = params['batch_size']
        nworkers = int(os.cpu_count() / 2)  # params['nworkers']
        transfer_out_path = osp.join(transferset_out_path, 'transferset.pickle')
        adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)

        print('=> constructing transfer set...')
        transferset = adversary.get_transferset(params['budget'])
        with open(transfer_out_path, 'wb') as wf:
            pickle.dump(transferset, wf)
        print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

        # Store arguments
        params['created_on'] = str(datetime.now())
        params_out_path = osp.join(transferset_out_path, 'params_transfer.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
