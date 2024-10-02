import torch
import torch.nn as nn
import torchvision
import lightning as L
from torchmetrics import Accuracy
from cifar10_resnext import model


class CIFAR10ResNeXt(L.LightningModule):
    def __init__(self, m):
        super().__init__()
        self.model = model.pick(m)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # lr function for resnext models
        def lr_fn(epochs):
            if epochs < 150:
                return 1
            elif epochs < 225:
                return 0.1
            else:
                return 0.01

        # lr function for wide resnet model
        def lr_fn2(epochs):
            if epochs < 60:
                return 1
            elif epochs < 120:
                return 0.8
            elif epochs < 160:
                return 0.6
            else:
                return 0.4

        # optimizer for resnext
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        # optimizer for wide resnet
        optimizer2 = torch.optim.SGD(self.model.parameters(), lr=1e-1, nesterov=True, momentum=0.9, weight_decay=5e-4)
        # lr scheduler for each models
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn),
            'name': 'learning_rate',
            'interval': 'epoch'
        }
        scheduler2 = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lr_fn2),
            'name': 'learning_rate',
            'interval': 'epoch'
        }
        if self.model == 'wresnet':
            return {"optimizer": optimizer2, "lr_scheduler": scheduler2}
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x_tr, y_tr = batch
        hypothesis = self.model(x_tr)
        loss = self.loss_fn(hypothesis, y_tr)
        correct_pred = torch.argmax(hypothesis, dim=1)
        acc = self.acc(correct_pred, y_tr)
        error_rate = (1 - acc)*100
        self.log("training loss", loss.item())
        self.log("training error rate", error_rate)

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        hypothesis = self.model(x_val)
        correct_pred = torch.argmax(hypothesis, dim=1)
        loss = self.loss_fn(hypothesis, y_val)
        acc = self.acc(correct_pred, y_val)
        error_rate = (1 - acc)*100
        self.log("validation loss", loss.item())
        self.log("validation error rate", error_rate)

    def test_step(self, batch, batch_idx):
        x_test, y_test = batch
        hypothesis = self.model(x_test)
        correct_pred = torch.argmax(hypothesis, dim=1)
        acc = self.acc(correct_pred, y_test)
        error_rate = (1 - acc)*100
        self.log("test error rate", error_rate)