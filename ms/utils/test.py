#!/usr/bin/python

import argparse
import os
import os.path as osp
import pickle
import time

import ms.config as cfg
import torch
import torch.nn as nn
import wandb
from ms import datasets
from ms.utils.model_pl import LitBlackBoxModule, LitBlackBoxNormModule
from torchmetrics.functional import accuracy, hamming_distance
from tqdm import tqdm


def test(model, model_dir, with_norm_layer_on_forward_only=False, with_norm_layer_layer_on_original_arch=False):
    """
    :param model:
    :param model_dir:
    :return: float(acc), preds, y_ref
    """
    batch_size = 32
    dataset_name = (model_dir[model_dir.rfind('/') + 1:].split('-')[0]).upper()
    print(f"TEST DATASET: {dataset_name}")
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    if with_norm_layer_on_forward_only or with_norm_layer_layer_on_original_arch:
        test_transform = datasets.modelfamily_to_transforms_without_normalization[modelfamily]['test']
    else:
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    dataset_test = dataset(train=False, transform=test_transform)
    print(f"Nb. of test samples : {len(dataset_test)}")
    idx_set = set(range(len(dataset_test)))
    preds = None

    y_ref = torch.LongTensor([dataset_test[i][1] for i in range(len(dataset_test))]).to(model.device)

    for b in tqdm(range(0, len(dataset_test), batch_size)):
        start = b
        end = min(b + batch_size, len(dataset_test))
        x_batch = torch.stack([dataset_test[i][0] for i in range(start, end)]).to(model.device)
        y_ref_batch = torch.LongTensor([dataset_test[i][1] for i in range(start, end)]).to(model.device)

        assert len(x_batch) == len(y_ref_batch)

        logits = model(x_batch)
        preds_batch = torch.argmax(logits, dim=1)
        if preds is None:
            preds = preds_batch
        else:
            preds = torch.cat([preds, preds_batch])

    assert len(preds) == len(y_ref)
    acc = accuracy(preds, y_ref)
    model.log(f"{model_dir[model_dir.rfind('/') + 1:]}_acc", acc)
    print(f"Test accuracy: {acc:.2%}")

    return float(acc), preds, y_ref


def test_custom_testset(model, model_dir, testset, description,
                        with_norm_layer_on_forward_only=False,
                        with_norm_layer_layer_on_original_arch=False, dataset_name=None):
    batch_size = 32
    if dataset_name is None:
        dataset_name = (model_dir[model_dir.rfind('/') + 1:].split('-')[0]).upper()
    print(f"TEST DATASET: {dataset_name}")
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    if with_norm_layer_on_forward_only or with_norm_layer_layer_on_original_arch:
        test_transform = datasets.modelfamily_to_transforms_without_normalization[modelfamily]['test']
    else:
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    dataset_test = dataset(train=False, transform=test_transform)
    print(f"Nb. of test samples : {len(dataset_test)}")
    print(f"Nb. of test samples : {len(testset)}")
    idx_set = set(range(len(dataset_test)))
    preds = None

    y_ref = torch.LongTensor([dataset_test[i][1] for i in range(len(testset))]).to(model.device)

    for b in tqdm(range(0, len(testset), batch_size)):
        start = b
        end = min(b + batch_size, len(testset))
        x_batch = torch.stack([torch.Tensor(testset[i][0]) for i in range(start, end)]).to(model.device)
        y_ref_batch = torch.LongTensor([dataset_test[i][1] for i in range(start, end)]).to(model.device)

        assert len(x_batch) == len(y_ref_batch)

        logits = model(x_batch)
        preds_batch = torch.argmax(logits, dim=1)
        if preds is None:
            preds = preds_batch
        else:
            preds = torch.cat([preds, preds_batch])

    assert len(preds) == len(y_ref)
    acc = accuracy(preds, y_ref[:len(preds)])
    model.log(f"{model_dir[model_dir.rfind('/') + 1:]}_acc", acc)

    print(f"Test accuracy {description}: {acc:.2%}")

    return float(acc), preds, y_ref


def get_extraction_metrics(y_victim, y_adv, y_ref):
    acc_victim = accuracy(y_victim, y_ref)
    acc_adv = accuracy(y_adv, y_ref)
    agreement = 1 - float(hamming_distance(y_adv, y_victim))
    extraction_ratio = acc_adv / acc_victim
    try:
        agreement2 = accuracy(y_adv, y_victim)  # TODO cehck if argmax is applied here
    except:
        agreement2 = -1
    extraction_metrics = dict(acc_victim=float(acc_victim),
                              acc_adv=float(acc_adv),
                              agreement=float((y_adv == y_victim).sum() / len(y_adv)),
                              extraction_ratio=float(extraction_ratio),
                              agreement2=float(agreement2))

    return extraction_metrics


def show_extraction_metrics(d):
    for key, val in d.items():
        print(f"|\t{key}: {val:.2%}", end='\t')
    print("|")


def get_model_checkpoint_path(model_dir, select=0):
    """

    :param model_dir: (str) directory of the model checkpoint
    :param select: (int) which model or models to select:
            * >=0 : index of model ordered by modification date, 0 being the most recent one
            * -1: the oldest model  trained
            * -2: all checkpoints (*.ckpt) available in the folder
    :return: one checkpoint (str) if select >= -1; otherwise multiple checkpoints  (List[str])
    """
    checkpoint_files = sorted([f"{model_dir}/{file}" for file in os.listdir(model_dir) if file.endswith('.ckpt')],
                              key=os.path.getmtime, reverse=True)
    table_len = 113
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f'No checkpoints in {model_dir}!')
    if len(checkpoint_files) > 1:
        print('MULTIPLE MODEL CHECKPOINTS'.center(table_len, '█'))
        print('-' * table_len)
        print(f'|\t{"Model path".center(table_len * 3 // 5)}\t|\t{"Last modified date".center(24)}\t|')
        print('-' * table_len)
        for model in checkpoint_files:
            print(f'|\t{model[model.find("/") + 1:]}\t|\t{time.ctime(os.path.getctime(model))}\t|')
        # print(checkpoint_files)
        print('-' * table_len)
    else:
        select = 0
    if select >= -1:
        checkpoint_path = checkpoint_files[select]
        print(f"Loading checkpoint model: {checkpoint_path}")
        print(f"Last modified: {time.ctime(os.path.getctime(checkpoint_path))}")
        return checkpoint_path
    return checkpoint_files


def load_model(ckpt_path, with_norm_layer_on_forward_only=False,
               with_norm_layer_layer_on_original_arch=False):
    print(f"Loading: {ckpt_path}")

    if with_norm_layer_on_forward_only:
        print("INFO: LitBlackBoxModuleNorm")
        blackbox = LitBlackBoxNormModule(ckpt_path)
    else:
        print("INFO: LitBlackBoxModule")
        blackbox = LitBlackBoxModule.load_from_checkpoint(checkpoint_path=ckpt_path)

    # # ------ It is important to freeze the model, otherwise it will overload the RAM
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if torch.cuda.device_count() > 1:
        blackbox = nn.DataParallel(blackbox)
    else:
        blackbox.to(device)
    # blackbox.to(device)
    blackbox.eval()
    blackbox.freeze()
    print(f"Inference on GPU: {blackbox.on_gpu} ({torch.cuda.get_device_name(0)})")
    return blackbox


def load_best_model(selected_checkpoints, blackbox_dir, with_norm_layer_on_forward_only=False
                    ):
    print(f'Finding the best from {selected_checkpoints}')
    best_acc = 0.0
    if isinstance(selected_checkpoints, str): selected_checkpoints = [selected_checkpoints]
    for ckpt_path in selected_checkpoints:
        blackbox = load_model(ckpt_path, with_norm_layer_on_forward_only
                              )

        # ----------- Initialize adversary

        acc, selected_y_hat, y_ref = test(blackbox, blackbox_dir, with_norm_layer_on_forward_only
                                          )
        if acc > best_acc:
            best_model = blackbox
            checkpoint_path = ckpt_path
            best_acc = acc
            y_hat = selected_y_hat
        del blackbox
    print(f'Selected: {checkpoint_path}\t(test accuracy: {best_acc:.2%})\n')
    return best_model, best_acc, y_hat, y_ref, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Testing victim models models')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.ckpt" and "params.json"')
    parser.add_argument('-a', '--adv_model_dir', type=str, default=None,
                        help='Path to adversary model. Should contain files "model_best.ckpt" and "params.json"')
    parser.add_argument('-v', '--victim_select', type=int, help='Which victim, last: -1 or best: -2', default=-1)
    parser.add_argument('-f', '--filter', type=str, help='filter for victim and adv name choice', default="")
    parser.add_argument('-n', '--adv_select', type=int, help='Nb of adversaries to show:  last: -1 or all: -2',
                        default=-1)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['victim_model_dir']

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Available device: {device}")

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    model_name = blackbox_dir[blackbox_dir.rfind('/') + 1:]

    dataset_name = (model_name.split('-')[0]).upper()
    adv_name = params['adv_model_dir'][params['adv_model_dir'].rfind('/') + 1:]
    run = wandb.init(project=cfg.WB_PROJECT, entity=cfg.WB_ENTITY,
                     name=f"extraction-{model_name}-{adv_name}{params['filter']}",
                     reinit=True,
                     job_type='extraction_test',
                     group=f"extraction-{dataset_name}", tags=['test', dataset_name])

    if params['filter'] == "":
        # load most recent victim
        selected_checkpoints = get_model_checkpoint_path(blackbox_dir, params['victim_select'])
    else:
        selected_checkpoints = sorted([f"{blackbox_dir}/{file}" for file in os.listdir(blackbox_dir)
                                       if file.endswith('.ckpt') and params['filter'] in file],
                                      key=os.path.getmtime, reverse=True)


    best_model, best_acc, y_hat, y_ref, checkpoint_path = load_best_model(selected_checkpoints, blackbox_dir, False)

    cm = wandb.plot.confusion_matrix(
        y_true=y_ref.cpu().numpy(),
        preds=y_hat.cpu().numpy(),
        # class_names=class_names
    )
    wandb.log({"conf_mat_victim": cm})

    cols = ['victim_model', 'knockoff_model', 'budget', 'acc_victim', 'acc_adv', 'agreement', 'extraction_ratio',
            'agreement2']
    table = wandb.Table(columns=cols)
    if params['adv_model_dir']:
        y_victim = y_hat
        if params['filter'] == "":
            # load most recent victim
            selected_checkpoints = get_model_checkpoint_path(params['adv_model_dir'], params['adv_select'])
        else:
            selected_checkpoints = sorted(
                [f"{params['adv_model_dir']}/{file}" for file in os.listdir(params['adv_model_dir'])
                 if file.endswith('.ckpt') and params['filter'] in file],
                key=os.path.getmtime, reverse=True)
        if isinstance(selected_checkpoints, str): selected_checkpoints = [selected_checkpoints]
        # print(selected_checkpoints)
        for ckpt_path in selected_checkpoints:
            dictio = {}
            print(f"Loading: {ckpt_path}")
            adversary = LitBlackBoxModule.load_from_checkpoint(checkpoint_path=ckpt_path)

            # # ------ It is important to freeze the model, otherwise it will overload the RAM
            if torch.cuda.device_count() > 1:
                adversary = nn.DataParallel(adversary)
                adversary.eval()
            else:
                adversary.to(device)
                adversary.eval()
                adversary.freeze()
            _, y_adv, _ = test(adversary, params['adv_model_dir'])
            del adversary
            extraction_metrics = get_extraction_metrics(y_victim, y_adv, y_ref)
            show_extraction_metrics(extraction_metrics)
            budget = ckpt_path[ckpt_path.rfind('-') + 1:-5]
            extraction_metrics['budget'] = int(budget) if budget.isdigit() else f"unk-{budget}"
            wandb.log(extraction_metrics)
            extraction_metrics['victim_model'] = checkpoint_path
            extraction_metrics['knockoff_model'] = ckpt_path
            cm = wandb.plot.confusion_matrix(
                y_true=y_ref.cpu().numpy(),
                preds=y_adv.cpu().numpy(),
                # class_names=class_names
            )
            # s[s.rfind('-')+1:-5]
            wandb.log({f"conf_mat_adv-{budget}": cm})
            table.add_data(*[extraction_metrics[col] for col in cols])
    wandb.log({"Table": table})

    print('=> constructing ref set...')
    refset = y_hat
    transfer_out_path = osp.join(out_path, 'refset.pickle')
    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(refset, wf)
    print('=> ref set ({} samples) written to: {}'.format(len(refset), transfer_out_path))


if __name__ == '__main__':
    main()
