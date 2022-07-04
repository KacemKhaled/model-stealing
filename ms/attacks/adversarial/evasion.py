# -*- coding: utf-8 -*-
"""
Adversarial Example Generation
==============================
Code adapted from : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

"""

from __future__ import print_function

import argparse
import json
import os.path as osp
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import ms.config as cfg
import ms.utils.utils as ms_utils
import numpy as np
import torch
import torchattacks
import wandb
from ms import datasets
from ms.utils.test import get_model_checkpoint_path, load_best_model, test_custom_testset
from torch.utils.data import DataLoader
from tqdm import tqdm

epsilons = [0, 0.01, 0.03, .05, .1, .15, .2, .25, .3]
show_plots = False

generate_test_adv = True
generate_train_adv = True

replace_test_adv = False
replace_train_adv = False
targeted = True


def adv_attack(model, modelfamily, device, data_loader, epsilons, attack='FGSM'):
    # Accuracy counter
    correct = 0
    adv_examples = []
    all_adv_examples = {eps: {'data': [], 'info': []} for eps in epsilons}
    all_accuracies = {eps: 0.0 for eps in epsilons}
    # Loop over all examples in test set
    c = 0
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        for eps in epsilons:
            if attack == 'fgsm':
                atk = torchattacks.FGSM(model, eps)
            elif attack == 'pgd':
                atk = torchattacks.PGD(model, eps, alpha=2 / 255, steps=4)
            else:
                raise ValueError(f'Attack {attack} not implemented!')
            adv_ex = atk(data, target)
            # Re-classify the perturbed image
            output = model(adv_ex)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                all_accuracies[eps] += 1
            adv_ex = adv_ex.squeeze().detach().cpu().numpy()
            all_adv_examples[eps]['data'].append(
                (adv_ex.squeeze(), target.cpu().squeeze()))
            all_adv_examples[eps]['info'].append(
                [init_pred.item(), final_pred.item(), torch.LongTensor(target.cpu()).item()])
        c += 1
    for eps in epsilons:
        pred = torch.LongTensor(all_adv_examples[eps]['info'][1])
        target = torch.LongTensor(all_adv_examples[eps]['info'][2])
        all_accuracies[eps] = all_accuracies[eps] / float(len(data_loader))
    show_accuracies_per_epsilon(all_accuracies)
    return all_accuracies, all_adv_examples


def plot_predictions(model, dataset,
                     dataset_labels, label_dict,
                     batch_size, grid_height, grid_width):
    def make_prediction(model=None, img_vector=[],
                        label_dict={}, top_N=3,
                        model_input_shape=None):
        if model:
            # get model input shape
            if not model_input_shape:
                model_input_shape = (1, 32, 32, 3)

            # get prediction
            prediction = model.predict(img_vector.reshape(model_input_shape))[0]

            # get top N with confidence
            labels_predicted = [label_dict[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
            confidence_predicted = np.sort(prediction)[::-1][:top_N]

            return labels_predicted, confidence_predicted

    if model:
        f, ax = plt.subplots(grid_width, grid_height)
        f.set_size_inches(12, 12)

        random_batch_indx = np.random.permutation(np.arange(0, len(dataset)))[:batch_size]

        img_idx = 0
        for i in range(0, grid_width):
            for j in range(0, grid_height):
                actual_label = label_dict.get(dataset_labels[random_batch_indx[img_idx]].argmax())
                preds, confs_ = make_prediction(model,
                                                img_vector=dataset[random_batch_indx[img_idx]],
                                                label_dict=label_dict,
                                                top_N=1)
                ax[i][j].axis('off')
                ax[i][j].set_title('Actual:' + actual_label[:10] + \
                                   '\nPredicted:' + preds[0] + \
                                   '(' + str(round(confs_[0], 2)) + ')')
                ax[i][j].imshow(dataset[random_batch_indx[img_idx]])
                img_idx += 1

        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0.4, hspace=0.55)


def show_accuracies_per_epsilon(results):
    print(f"|\tEps\t|\tAcc\t|")
    for eps, acc in results.items():
        print(f"|\t{eps}\t|\t{acc:.2%}\t|")


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot(epsilons, accuracies, examples, modelfamily, name="", **params):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies.values(), "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    filename1 = f'{modelfamily}-Accuracy_vs_Epsilon-{datetime.now().strftime("%H_%M_%d_%m_%Y")}'
    path1 = osp.join(cfg.CACHE_ROOT, f"{filename1}.png")
    fig.savefig(path1, dpi=fig.dpi)

    data = [[x, y, z] for (x, y, z) in zip(epsilons, list(accuracies.values()), [name] * len(epsilons))]
    print(f"data: {data}")
    table = wandb.Table(data=data, columns=["Epsilon", "Accuracy", "Path"])
    try:
        wandb.log({f"Accuracy_vs_Epsilon-{name}": plt})
    except:
        print('Cannot plot {Accuracy_vs_Epsilon}')

    try:
        wandb.log({f"{modelfamily}-Accuracy_vs_Epsilon-{name}": wandb.plot.line(table, "Epsilon", "Accuracy",
                                                                                title="Accuracy Vs Epsilon")})
    except:
        print('Cannot plot {Accuracy vs Epsilon Table}')

    cnt = 0

    fig = plt.figure(figsize=(8, 10))
    nb_examples = 5
    try:
        table = wandb.Table(columns=['Epsilon', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'])
        for eps in epsilons:
            images = []
            labels = []
            for j in range(nb_examples):
                cnt += 1
                plt.subplot(len(epsilons), nb_examples, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(eps), fontsize=14)
                ex, _ = examples[eps]['data'][j]
                orig, adv, target = examples[eps]['info'][j]
                title = "Pred: {} -> Adv pred: {}\nTarget: {}".format(orig, adv, target)
                plt.title(title)
                # ex = ex.squeeze()

                if modelfamily == "mnist":
                    plt.imshow(ex, cmap="gray")
                else:

                    ex = ex.transpose(1, 2, 0)
                    plt.imshow(ex)

                images.append(ex)
                labels.append(title)

            table.add_data(eps, *[wandb.Image(image, caption=label) for image, label in zip(images, labels)])
            wandb.log({f"Epsilon {eps}": [wandb.Image(image, caption=label) for image, label in zip(images, labels)]})

        wandb.log({"Table": table})
        wandb.log({"Images": plt})

        plt.tight_layout()
        if show_plots:
            plt.show()
        filename2 = f'{modelfamily}-Adversarial_Examples-{datetime.now().strftime("%H_%M_%d_%m_%Y")}'
        path2 = osp.join(cfg.CACHE_ROOT, f"{filename2}.png")
        fig.savefig(path2, dpi=fig.dpi)

        wandb.log({f"{filename1}": wandb.Image(path1),
                   f"{filename2}": wandb.Image(path2)})
    except:
        print('Could not plot adv. examples table')


def save_data(data, epsilons, filename, out_path):
    for eps in epsilons:
        data_out_path = osp.join(out_path, f'{filename}-{eps}.pickle')
        with open(data_out_path, 'wb') as wf:
            pickle.dump(data[eps]['data'], wf)
        print('=> adversarial set ({} samples) written to: {}'.format(len(data[eps]['data']), data_out_path))


def load_adversarial_examples(epsilons, filename_prefix, out_path):
    data = {}
    loaded_epsilons = []

    for eps in epsilons:
        data_out_path = osp.join(out_path, f'{filename_prefix}-{eps}.pickle')
        if osp.exists(data_out_path):

            with open(data_out_path, 'rb') as rf:
                try:
                    data[eps] = pickle.load(rf)
                    loaded_epsilons.append(eps)
                except:
                    print(f'Cannot load file (eps: {eps}): {data_out_path}')
    unloaded_epsilons = [eps for eps in epsilons if eps not in loaded_epsilons]
    print(
        f"INFO: found {len(loaded_epsilons)} adversarial files in {out_path}/{filename_prefix}** Epsilons {loaded_epsilons}")
    print(f"INFO: Epsilons not found {len(unloaded_epsilons)}: ({unloaded_epsilons})")
    return data, loaded_epsilons, unloaded_epsilons


def main():
    # [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb
    use_cuda = True

    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('attack', metavar='PI', type=str, help='Policy to use while training', default='fgsm',
                        choices=['fgsm', 'pgd'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.ckpt" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str, default='adv_data',
                        help='Destination directory to store adversarial set', required=False)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    parser.add_argument('-v', '--victim_select', type=int, help='Which victim, last: -1 or best: -2', default=0)
    parser.add_argument('--epsilons', metavar='EPS', type=str,
                        help='Comma separated values of epsilons. Adversarial examples will be generated for each epsilon.')

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)
    epsilons = [float(b) for b in params['epsilons'].split(',')]

    out_path = osp.join(params['victim_model_dir'], params['out_dir'])
    # TODO create folder to store output adversarial examples
    ms_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Available device: {device}")

    # TODO load the model here:
    model_dir = params['victim_model_dir']

    # load most recent victim
    selected_checkpoints = get_model_checkpoint_path(model_dir, 0)
    # here we add a norm layer on forward
    model, test_acc, _, _, checkpoint_path = load_best_model(selected_checkpoints, model_dir,
                                                             with_norm_layer_on_forward_only=True)

    batch_size = 32
    dataset_name = (model_dir[model_dir.rfind('/') + 1:].split('-')[0]).upper()
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
    model_name = f"{model_dir[model_dir.rfind('/') + 1:]}-{params['attack']}-{str_epsilons}"

    run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY, name=model_name,
                     # id=model_name, ## no id here since the name is long
                     reinit=True, group='Adversarial_Examples')  # , id=model_name)

    filename_prefix = f'{dataset_name}-test-{params["attack"]}'
    data_test, loaded_test_epsilons, unloaded_test_epsilons = load_adversarial_examples(epsilons, filename_prefix,
                                                                                        out_path)
    dict_acc = {}
    if loaded_test_epsilons:
        for eps in loaded_test_epsilons:
            # model.test()
            description = f"(Attack: {params['attack']}, Eps: {eps})"
            adv_test_set = ms_utils.samples_to_adv_data_set(data_test[eps], transform=test_transform)
            adv_acc, adv_preds, y_ref = test_custom_testset(model, model_dir, adv_test_set, description,
                                                            with_norm_layer_on_forward_only=True,
                                                            with_norm_layer_layer_on_original_arch=False)
            dict_acc[eps] = adv_acc

            wandb.log({f"adv_acc - attack {params['attack']}": adv_acc, 'eps': eps})
        plot(loaded_test_epsilons, dict_acc, data_test, dataset_name)

    if generate_test_adv:
        if replace_test_adv:
            print(f"Warning: replacing already existing adversarial examples {unloaded_test_epsilons} with {epsilons}")
            unloaded_test_epsilons = epsilons[:]
        if len(unloaded_test_epsilons) > 0:
            print(f"Info: generating adversarial examples for epsilons {unloaded_test_epsilons}")
            accuracies, test_examples = adv_attack(model, dataset_name, device, test_loader, unloaded_test_epsilons,
                                                   attack=params['attack'])
            save_data(test_examples, unloaded_test_epsilons, filename_prefix, out_path)
            plot(unloaded_test_epsilons, accuracies, test_examples, modelfamily)

    filename_prefix = f'{dataset_name}-train-{params["attack"]}'
    data_train, loaded_train_epsilons, unloaded_train_epsilons = load_adversarial_examples(epsilons, filename_prefix,
                                                                                           out_path)

    if generate_train_adv:
        if replace_train_adv:
            print(f"Warning: replacing already existing adversarial examples {unloaded_train_epsilons} with {epsilons}")
            unloaded_train_epsilons = epsilons[:]
        if len(unloaded_train_epsilons) > 0:
            print(f"Info: generating adversarial examples for epsilons {unloaded_train_epsilons}")
            _, train_examples = adv_attack(model, modelfamily, device, train_loader, unloaded_train_epsilons,
                                           attack=params['attack'])
            save_data(train_examples, unloaded_train_epsilons, f'{dataset_name}-train-{params["attack"]}', out_path)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, f'params_advs_{params["created_on"]}.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
