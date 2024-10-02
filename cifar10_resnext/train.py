import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from cifar10_resnext.modlit import CIFAR10ResNeXt
from cifar10_resnext.data import CIFAR10DataModule
from torchsummary import summary
import lightning as L
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.wresnet / resnext29)')
    m = parser.parse_args().model

    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)
    dm.prepare_data()
    dm.setup(stage="fit")

    if m == 'wresnet':
        total_epochs = 200
    else:
        total_epochs = 300

    print(f'getting model {m}, total_epochs: {total_epochs}')

    net = CIFAR10ResNeXt(m)
    # print(net)
    # summary(net.to('cuda'), (3, 32, 32))
    # exit(0)

    wandb_logger = WandbLogger(log_model=False, name=f'{m}', project='cifar10_resnext')
    cp_callback = ModelCheckpoint(
        dirpath='./model',
        filename=f'model_{m}',
        mode='min',
        monitor='validation loss',
        save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        max_epochs=total_epochs,
        callbacks=[cp_callback, lr_monitor],
        logger=wandb_logger,
        accelerator='cuda',
        devices=1
    )
    trainer.fit(net, dm)


if __name__ == '__main__':
    main()
