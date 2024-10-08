import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cifar10_resnext.modlit import CIFAR10ResNeXt
from cifar10_resnext.data import CIFAR10DataModule
import lightning as L
import argparse


def main():
    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.wresnet / resnext29)')
    m = parser.parse_args().model

    # assign data module
    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)

    # load checkpoint
    checkpoint = f"./model/model_{m}.ckpt"
    net = CIFAR10ResNeXt.load_from_checkpoint(checkpoint, m=m)
    trainer = L.Trainer(accelerator='cuda', devices=1)
    # test
    trainer.test(net, dm)


if __name__ == '__main__':
    main()
