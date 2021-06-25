#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np


def data_load():
    pre_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=pre_processing)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return test_ds, classes

def get_accuracy(y, label):
    y_idx = torch.argmax(y, dim=1)
    result = y_idx - label

    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1
    return num_correct/y.shape[0]

def test(dataloader, model, criterion, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        test_loss_list, test_acc_list = [], []

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            correct = model(x)
            test_loss = criterion(correct, y)

            test_acc = get_accuracy(correct, y)

            test_loss_list.append(test_loss.item())
            test_acc_list.append(test_acc)
        print(f'Test loss: {np.mean(test_loss_list):.4f} | Test acc: {np.mean(test_acc_list):.4f}')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    PATH1 = './models/cifar10.pth'
    PATH2 = './parameters/cifar10.pth'

    model = torch.load(PATH1).to(device)
    model.load_state_dict(torch.load(PATH2))

    batch_size = 128
    
    test_ds, classes = data_load()
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    
    test(test_dl, model, criterion, device)

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()