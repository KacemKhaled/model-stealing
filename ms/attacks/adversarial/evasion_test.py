# -*- coding: utf-8 -*-
"""
Adversarial Example Generation
==============================
Code adapted from : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

"""

from __future__ import print_function

import argparse
import os

import ms.config as cfg
import ms.utils.utils as ms_utils
import torch
import wandb
from ms import datasets
from ms.attacks.adversarial.evasion import load_adversarial_examples
from ms.utils.test import test_custom_testset, load_model
from torch.utils.data import DataLoader


def plot(epsilons, accuracies, dataset_name, name=""):
    data = [[x, y, z] for (x, y, z) in zip(epsilons, list(accuracies.values()), [name] * len(epsilons))]
    print(f"data: {data}")
    table = wandb.Table(data=data, columns=["Epsilon", "Accuracy", "Path"])
    try:
        wandb.log({f"{dataset_name}-Accuracy_vs_Epsilon-": wandb.plot.line(table, "Epsilon", "Accuracy",
                                                                           title=f"{name}")})
    except:
        print('Cannot plot {Accuracy vs Epsilon Table}')


def main():
    use_cuda = True

    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('attacks', metavar='PI', type=str, help='attacks used for testing adversarial robustness',
                        default='fgsm')
    parser.add_argument('models_dir', metavar='PATH', type=str, default=cfg.MODEL_DIR,
                        help='Path to models. Should contain files "model_best.ckpt"')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Model dataset', default=None)
    parser.add_argument('--adv_data_path', metavar='PATH', type=str, default='adv_data',
                        help='Destination directory to store adversarial set', required=False)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--epsilons', metavar='EPS', type=str,
                        help='Comma separated values of epsilons. Adversarial examples will be generated for each epsilon.')
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('-f', '--filter', type=str, help='filter for victim and adv name choice', default="")
    args = parser.parse_args()
    params = vars(args)
    epsilons = [float(b) for b in params['epsilons'].split(',')]

    adv_data_path = params['adv_data_path']

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Available device: {device}")

    # TODO load the model here:
    model_dir = params['models_dir']

    batch_size = 32
    dataset_name = params['dataset_name']
    print(f"TEST DATASET: {dataset_name}")
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    # We do not need to perform any transformation to the data
    test_transform = datasets.modelfamily_to_transforms_without_normalization[modelfamily]['test']

    dataset_train = dataset(train=True, transform=test_transform)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=False)
    print(f"Nb. of train samples : {len(dataset_train)}")

    dataset_test = dataset(train=False, transform=test_transform)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    print(f"Nb. of test samples : {len(dataset_test)}")
    str_epsilons = '-'.join([str(eps) for eps in epsilons])
    group_name = f"{dataset_name}-{params['attacks']}"

    # load most recent victim
    # here we add a norm layer on forward
    models_ckpt_paths = []

    for root, subdirs, files in os.walk(model_dir):
        models_ckpt_paths += [os.path.join(root, file)
                              for file in files
                              if dataset_name in root and (file.endswith('best.ckpt') or \
                                                           file.endswith('fgsm-0.03.ckpt') or file.endswith(
                        'fgsm-0.05.ckpt') or \
                                                           file.endswith('pgd-0.03.ckpt') or file.endswith(
                        'pgd-0.05.ckpt') or \
                                                           file.endswith('50000.ckpt'))]
    # check if files are there !
    models_ckpt_paths.sort(reverse=True)
    print(f"INFO: Found {len(models_ckpt_paths)} models to test for adversarial robustness: ")
    print("\n".join(models_ckpt_paths))

    attacks = [b for b in params['attacks'].split(',')]
    epsilons_cols = [f'Eps: {eps}' for eps in epsilons]
    columns = ['Model'] + epsilons_cols

    i = 1
    for selected_ckpt in models_ckpt_paths:
        print('-*' * 20)
        print(f'INFO: model {i}/{len(models_ckpt_paths)}')
        model = load_model(selected_ckpt,  # was load_best_model ()
                           with_norm_layer_on_forward_only=True)
        log_name = selected_ckpt[selected_ckpt.find('models/') + len('models/'):]
        accuracies_per_attack = {}
        for attack in attacks:

            filename_prefix = f'{dataset_name}-test-{attack}'
            log_name_wandb = f"{filename_prefix}: {log_name}"

            run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY, name=log_name_wandb,
                             job_type=f'adversarial_robustness_{attack}',
                             # id=model_name, ## no id here since the name is long
                             reinit=True, group=group_name,
                             tags=[dataset_name, 'adv exps'])  # , id=model_name)
            data_test, loaded_test_epsilons, unloaded_test_epsilons = load_adversarial_examples(epsilons,
                                                                                                filename_prefix,
                                                                                                adv_data_path)
            dict_acc = {}
            table = wandb.Table(columns=columns)

            if loaded_test_epsilons:
                for eps in loaded_test_epsilons:
                    # model.test()
                    description = f"(Attack: {attack}, Eps: {eps})"
                    adv_test_set = ms_utils.samples_to_adv_data_set(data_test[eps], transform=test_transform)
                    adv_acc, adv_preds, y_ref = test_custom_testset(model, model_dir, adv_test_set, description,
                                                                    with_norm_layer_on_forward_only=True,
                                                                    with_norm_layer_layer_on_original_arch=False,
                                                                    dataset_name=dataset_name)
                    dict_acc[eps] = adv_acc

                    wandb.log({f"adv_acc": adv_acc, 'eps': eps})
                data = [log_name_wandb] + list(dict_acc.values())
                table.add_data(*data)

                accuracies_per_attack[attack] = list(dict_acc.values())
            wandb.log({"table": table})

        del model
        i += 1


if __name__ == '__main__':
    main()
