#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import torch
from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from VOCdownloader import VOCDownLoader as VOC

from typing import Dict


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    return img

def data_show(_data):
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(_data), size=(1,)).item()
        image, target = _data[sample_idx]
        #label = target['annotation']['object'][0]['name']
        plt.subplot(cols, rows, i)
        img = imshow(image)
        plt.imshow(img)
        #plt.title(f'Class: {label}')
        plt.axis("off")
    plt.show()

def draw_bbox(_data):
    image, target = _data[10]
    image = imshow(image)
    print(image.size, target)
    print(target['annotation']['object'])

    obj: Dict = {}
    for i in range(len(target['annotation']['object'])):
        name = target['annotation']['object'][i]['name']
        bbox = target['annotation']['object'][i]['bndbox']
        for key, val in bbox.items():
            bbox[key] = int(val)
        obj[i] = [name, bbox]
    print(obj)

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i in range(len(obj)):
        xmin = obj[i][1]['xmin']
        ymin = obj[i][1]['ymin']
        xmax = obj[i][1]['xmax']
        ymax = obj[i][1]['ymax']

        bbox = patches.Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin), linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(bbox)
        ax.text(xmin, ymin, obj[i][0], size=14, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.savefig('./image/04.png')
    plt.show()

def main():
    train = VOC(root='./data', year='2007', image_set='train')
    print(train.root, train.year, train.image_set)

    pre_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    _dataset = datasets.VOCDetection(root=train.root, year=train.year, image_set=train.image_set, transform=pre_processing)
    train_len = int(len(_dataset) * 0.8)
    valid_len = int(len(_dataset) * 0.2) + 1
    train_data, valid_data = torch.utils.data.random_split(_dataset, [train_len, valid_len])
    
    #data_show(train_data)
    draw_bbox(train_data)

if __name__ == "__main__":
    main()