from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from ms import datasets
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy


class ConvNet(nn.Module):
    # source: https://github.com/jamespengcheng/PyTorch-CNN-on-CIFAR10
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = self.pool(x) 
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x)) 
        x = self.pool(x) 
        x = self.Dropout(x)
        x = x.view(-1, 8 * 8 * 256) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


class Net(nn.Module):
    """A simple CIFAR-10 network

        Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def add_conv_stage(dim_in, dim_out, kernel_size=(3, 3), stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU()
    )


# LeNet Model definition
class CNNMnist2(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNMnist(nn.Module):
    'PyTorch MNIST Example'

    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class genericDataModule(VisionDataModule):
    def __init__(self, dataset_name, val_split=0.2, batch_size=32, num_workers=6, pin_memory=False):
        # TODO check utility of data_dir argument, seems like it is not used -- DELETED arg
        super().__init__()
        self.dataset_name = dataset_name
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        dataset_name = self.dataset_name
        valid_datasets = datasets.__dict__.keys()
        if dataset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset = datasets.__dict__[dataset_name]

        dataset_train = dataset(train=True)
        splits = self._get_splits(len(dataset_train))
        self.num_classes = len(dataset().classes)
        print(f"------------------- CLASSES: {self.num_classes} - DATASET SPLITS: {splits}")
        self.steps_per_epoch = ceil(splits[0] / self.batch_size)

    def prepare_data(self):
        # ----------- Set up dataset
        dataset_name = self.dataset_name
        valid_datasets = datasets.__dict__.keys()
        if dataset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset = datasets.__dict__[dataset_name]

    def setup(self, stage: str):
        dataset = datasets.__dict__[self.dataset_name]
        modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
            dataset_train = dataset(train=True, transform=train_transform)
            self.dataset_train = self._split_dataset(dataset_train, train=True)
            self.dataset_val = self._split_dataset(dataset_train, train=False)
            print(f"Train set : {len(self.dataset_train)}\nValidation set: {len(self.dataset_val)}")

        if stage == "test" or stage is None:
            test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
            self.dataset_test = dataset(train=False, transform=test_transform)
            print(f"Test set : {len(self.dataset_test)}")


class transfersetDataModule(VisionDataModule):
    def __init__(self, transferset, testset_name, val_split, batch_size, num_workers, params,
                 pin_memory=False):
        super().__init__()
        self.transferset = transferset
        self.testset_name = testset_name
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.params = params
        self.pin_memory = pin_memory
        valid_datasets = datasets.__dict__.keys()
        if self.testset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        splits = self._get_splits(len(transferset))
        print(f"-------- Transferset SPLITS: {splits}")
        self.steps_per_epoch = ceil(splits[0] / self.batch_size)

    def prepare_data(self):
        # ----------- Set up testset
        dataset_name = self.testset_name
        valid_datasets = datasets.__dict__.keys()
        if dataset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset = datasets.__dict__[dataset_name]
        testset = dataset(train=False)

    def setup(self, stage: str):
        dataset = datasets.__dict__[self.testset_name]
        modelfamily = datasets.dataset_to_modelfamily[self.testset_name]
        if stage == "fit" or stage is None:
            dataset_train = self.transferset
            # Split
            self.dataset_train = self._split_dataset(dataset_train, train=True)
            self.dataset_val = self._split_dataset(dataset_train, train=False)
            print(f"Train set : {len(self.dataset_train)}\nValidation set: {len(self.dataset_val)}")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
            self.dataset_test = dataset(train=False, transform=test_transform)
            print(f"Test set : {len(self.dataset_test)}")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def load_DataModule(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, percent_validation=0.1, dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        normalization = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                             std=(0.2023, 0.1994, 0.2010))
    else:
        normalization = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            normalization,  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        val_split=percent_validation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    return cifar10_dm


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def add_norm_layer(model, dataset):
    print("INFO: adding a normalization layer")
    if 'MNIST' in dataset.upper():
        norm_layer = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    elif 'CIFAR' in dataset.upper():
        norm_layer = transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.247, 0.244, 0.262))
    else:
        norm_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return nn.Sequential(norm_layer, model)


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # feature_extract: Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params

    model_ft = None
    input_size = 0

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    valid_models = sorted(name for name in models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(models.__dict__[name]))
    assert model_name in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)

    model_ft = models.__dict__[model_name](pretrained=use_pretrained)  # , num_classes=num_classes)

    if model_name.startswith("resnet"):
        """ Resnet18
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("alexnet"):
        """ Alexnet
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("vgg"):
        """ VGG11_bn
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("squeezenet"):
        """ Squeezenet
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name.startswith("densenet"):
        """ Densenet
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("inception"):
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, using default settings...")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    return model_ft


def create_model(modelname="resnet18", num_classes=10, pretrained=False):
    print(f'INFO: CREATING MODEL {modelname}')
    """
    Modify the pre-existing Resnet architecture from TorchVision.
    The pre-existing architecture is based on ImageNet images (224x224) as input.
    So we need to modify it for CIFAR10 images (32x32).
    """
    # valid_models = ms.models.imagenet.__dict__.keys()
    valid_models = sorted(name for name in models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(models.__dict__[name]))
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    # Print the model we just instantiated
    model = models.__dict__[modelname](pretrained=pretrained, num_classes=num_classes)  
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    return model


def get_mnist_model(modelname='model'):
    valid_mnist_models = {
        'cnn': CNNMnist(),
        'lenet': LeNet()
    }
    assert modelname in valid_mnist_models, 'Model not recognized for the MNIST Dataset, Supported models = {}'.format(
        valid_mnist_models)
    model = valid_mnist_models[modelname]

    return model


class LitModule(LightningModule):
    def __init__(self, learning_rate, steps_per_epoch, params, num_classes):
        super().__init__()
        self.params = params
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        if 'MNIST' in self.params['dataset']:
            self.model = get_mnist_model(self.params['arch'])
        elif self.params['pretrained']:
            self.model = initialize_model(self.params['arch'], num_classes, feature_extract=False,
                                          use_pretrained=self.params['pretrained'])
        else:
            self.model = create_model(self.params['arch'], num_classes, pretrained=self.params['pretrained'])
        self.num_classes = num_classes

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)  # F.softmax(out, dim=1)#

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("train_loss", loss, on_epoch=True)  
        self.log("train_acc", acc, on_epoch=True)  
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        metrics = {'loss': loss, 'acc': acc}
        return metrics

    def evaluate(self, batch, stage=None, *args):
        if len(args) > 0:
            suffix = '_adv' if len(args[0]) > 0 and args[0][0] != 0 else ''
        else:
            suffix = ''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss{suffix}", loss, prog_bar=True, on_epoch=True)  
            self.log(f"{stage}_acc{suffix}", acc, prog_bar=True, on_epoch=True)  
            metrics = {f'{stage}_loss{suffix}': loss, f"{stage}_acc{suffix}": acc}
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.evaluate(batch, "val")
        return metrics

    def test_step(self, batch, batch_idx, *args):
        metrics = self.evaluate(batch, "test", args)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        # steps_per_epoch = 45000 // BATCH_SIZE
        scheduler = {
            'step': {
                "scheduler": StepLR(
                    optimizer=optimizer,
                    step_size=self.params['lr_step'],
                    gamma=self.params['lr_gamma'],
                    last_epoch=-1
                ),
                "interval": "epoch",
            },
            'onecycle': {
                "scheduler": OneCycleLR(
                    optimizer=optimizer,
                    max_lr=0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=self.steps_per_epoch,
                    last_epoch=-1
                ),
                "interval": "step",  # or 'epoch'
            }
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler[self.params['scheduler_choice']]}


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


class LitModuleAdversary(LightningModule):
    # Special Lightning module used as adversary for substitute model training for the Knockoff attack
    # The difference is this one can be trained with soft labels rather than hard labels.
    def __init__(self, learning_rate, steps_per_epoch, params, num_classes):
        super().__init__()
        self.params = params
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        # self.learning_rate = params['lr']
        self.save_hyperparameters()
        if 'MNIST' in self.params['dataset']:
            self.model = get_mnist_model(self.params['arch'])
        elif self.params['pretrained']:
            self.model = initialize_model(self.params['arch'], num_classes, feature_extract=False,
                                          use_pretrained=self.params['pretrained'])
        else:
            self.model = create_model(self.params['arch'], num_classes, pretrained=self.params['pretrained'])
        self.num_classes = num_classes

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)  # F.softmax(out, dim=1)#

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if not self.params['argmaxed']:
            # y = y.to(torch.long)
            # logits, y = logits.to(torch.long), y.to(torch.long)
            criterion_train = soft_cross_entropy
            loss = criterion_train(logits, y)
        else:
            loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        if len(y.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = y.max(1)
        else:
            target_labels = y

        acc = accuracy(preds, target_labels)
        self.log("train_loss", loss, on_epoch=True)  
        self.log("train_acc", acc, on_epoch=True)  
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        metrics = {'loss': loss, 'acc': acc}
        return metrics

    def evaluate(self, batch, stage=None):
        # print( batch)
        x, y = batch
        logits = self(x)

        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        if len(y.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = y.max(1)
        else:
            target_labels = y
        acc = accuracy(preds, target_labels)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)  
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)  
            metrics = {f'{stage}_loss': loss, f"{stage}_acc": acc}
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.evaluate(batch, "val")
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.evaluate(batch, "test")
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        print('-' * 50)
        print(self.hparams.learning_rate)
        print(optimizer.__repr__())

        scheduler = {
            'step': {
                "scheduler": StepLR(
                    optimizer=optimizer,
                    step_size=self.params['lr_step'],
                    gamma=self.params['lr_gamma'],
                    last_epoch=-1
                ),
                "interval": "epoch",
            },
            'onecycle': {
                "scheduler": OneCycleLR(
                    optimizer=optimizer,
                    max_lr=0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=self.steps_per_epoch,
                    last_epoch=-1
                ),
                "interval": "step",  # or 'epoch'
            }
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler[self.params['scheduler_choice']]}


class LitBlackBoxModule(LightningModule):
    """
    blackbox model for inference only
    """

    def __init__(self, learning_rate, steps_per_epoch, params, num_classes):
        super().__init__()
        self.params = params
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        if 'MNIST' in self.params['dataset']:
            self.model = get_mnist_model(self.params['arch'])
        elif self.params['pretrained']:
            self.model = initialize_model(self.params['arch'], num_classes, feature_extract=False,
                                          use_pretrained=self.params['pretrained'])
        else:
            self.model = create_model(self.params['arch'], num_classes, pretrained=self.params['pretrained'])
        self.num_classes = num_classes

    def forward(self, x):
        out = self.model(x)
        return F.softmax(out, dim=1)  # F.log_softmax(out, dim=1)#

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None


class LitBlackBoxNormModule(LightningModule):
    """
    a Black box model where we add a normalization layer for inference
    """

    def __init__(self, path):
        super().__init__()
        self.model = LitBlackBoxModule.load_from_checkpoint(path)
        if 'MNIST' in path.upper():
            self.norm_layer = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        elif 'CIFAR' in path.upper():
            self.norm_layer = transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.247, 0.244, 0.262))
        else:
            self.norm_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.norm_layer(x)
        out = self.model(x)
        return F.softmax(out, dim=1)  # F.log_softmax(out, dim=1)#

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

