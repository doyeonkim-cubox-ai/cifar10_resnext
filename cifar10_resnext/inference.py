import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cifar10_resnext.modlit import CIFAR10ResNeXt
import argparse
import numpy as np
from PIL import Image


def main():

    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.wresnet / resnext29)')
    m = parser.parse_args().model

    # load trained model
    checkpoint = f"./model/model_{m}.ckpt"
    model = CIFAR10ResNeXt.load_from_checkpoint(checkpoint, m=m)

    #########################################################################################################
    # ========================================== data preprocess ========================================== #
    #########################################################################################################
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    ])

    # using your own data
    parser.add_argument('-img', type=str, help='Input Image Path')
    img_path = parser.parse_args().img
    img = Image.open(img_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    # using random train data
    # test = torchvision.datasets.CIFAR10("./cifar10", train=True, transform=transform)
    # test_loader = DataLoader(test, batch_size=4, shuffle=True, num_workers=8)
    # data_iter = iter(test_loader)
    # img, label = next(data_iter)
    # print('GroundTruth: ', ' '.join('%5s' % classes[label[j]] for j in range(4)))
    # img_tensor = img.to('cuda')

    #########################################################################################################
    # ============================================== process ============================================== #
    #########################################################################################################
    # put inference img tensor into the prediction model
    result = model(img_tensor)

    #########################################################################################################
    # ============================================ postprocess ============================================ #
    #########################################################################################################
    # using your own data
    result = result.squeeze(0).detach().to('cpu').numpy()
    res = np.argmax(result)
    print(result)
    print(classes[res])
    # using random train data
    # _, predicted = torch.max(result, 1)
    #
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                               for j in range(4)))


if __name__ == '__main__':
    main()
