import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from utils.dataloader import get_dataloader
from utils.dataset import get_dataset

import matplotlib.pyplot as plt

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 64


# transforms with albumentations 
# find_lr coupled with one_cycle lr


class BasicBlock(LightningModule):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomBlock(LightningModule):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()

        self.inner_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.res_block = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.inner_layer(x)
        r = self.res_block(x)

        out = x + r

        return out


class CustomResNet(LightningModule):
    def __init__(self, num_classes=10,data_dir=PATH_DATASETS, hidden_size=16, lr=2e-4):
        super(CustomResNet, self).__init__()

        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = lr

        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.lr_change = []
        # self.outputs=[]
        self.train_step_losses = []
        self.train_step_acc = []

        self.val_step_losses = []
        self.val_step_acc = []

        test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

        self.accuracy = Accuracy(task='multiclass',num_classes=num_classes)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.prep_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer_1 = CustomBlock(in_channels=64, out_channels=128)

        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer_3 = CustomBlock(in_channels=256, out_channels=512)

        self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=4))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).cpu().float().mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.train_step_acc.append(acc)
        self.train_step_losses.append(loss.cpu().item())
        
        return {'loss':loss, 'train_acc': acc}

    def on_train_epoch_end(self):
        # batch_losses = [x["train_loss"] for x in outputs] #This part
        epoch_loss = sum(self.train_step_losses)/len(self.train_step_losses)
        # batch_accs =  [x["train_acc"] for x in outputs]   #This part
        epoch_acc = sum(self.train_step_acc)/len(self.train_step_acc)
        self.log("train_loss_epoch", epoch_loss, prog_bar=True)
        self.log("train_acc_epoch", epoch_acc, prog_bar=True)
        self.train_acc.append(epoch_acc)
        self.train_losses.append(epoch_loss)
        self.lr_change.append(self.scheduler.get_last_lr()[0])
        self.train_step_losses.clear()
        self.train_step_acc.clear()
        return epoch_acc
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).cpu().float().mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_step_acc.append(acc)
        self.val_step_losses.append(loss.cpu().item())
        return {'val_loss':loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        # batch_losses = [x["val_loss"] for x in outputs] #This part
        epoch_loss = sum(self.val_step_losses)/len(self.val_step_losses)
        # batch_accs =  [x["val_acc"] for x in outputs]   #This part
        epoch_acc = sum(self.val_step_acc)/len(self.val_step_acc)
        self.log("val_loss_epoch", epoch_loss, prog_bar=True)
        self.log("val_acc_epoch", epoch_acc, prog_bar=True)
        self.test_acc.append(epoch_acc)
        self.test_losses.append(epoch_loss)
        self.val_step_losses.clear()
        self.val_step_acc.clear()
        return epoch_acc

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.learning_rate,epochs=30,steps_per_epoch=len(self.cifar_full)//BATCH_SIZE)
        lr_scheduler = {'scheduler': self.scheduler, 'interval': 'step'}
        return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler}

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_full = get_dataset()[0]
            self.cifar_train, self.cifar_val = random_split(self.cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            self.cifar_test = get_dataset()[1]

    def train_dataloader(self):
        cifar_full = get_dataset()[0]
        return DataLoader(cifar_full, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
        # return get_dataloader()[0]


    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
        # return get_dataloader()[1]

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
        # return get_dataloader()[1]

    def draw_graphs(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")

    def draw_graphs_lr(self):
        # fig, axs = plt.subplots(1,1,figsize=(15,10))
        plt.plot(self.lr_change)