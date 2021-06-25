#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import CNNClassifier


def data_load():
    pre_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=pre_processing)

    train_len = int(len(ds) * 0.85)
    valid_len = int(len(ds) * 0.15)
    train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_ds, valid_ds, classes

def get_accuracy(y, label):
    y_idx = torch.argmax(y, dim=1)
    result = y_idx - label

    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1
    return num_correct/y.shape[0]

def train(dataloader, model, criterion, optimizer):
    num_batches = len(dataloader)
    train_loss_list, train_acc_list = [], []

    start_time = time.time()
    for batch, (x, y) in enumerate(dataloader):
        model.train()
        # x: 입력, y: 정답(레이블)을 받아온 후 device에 올려줌
        x, y = x.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(x)
        loss = criterion(pred, y)  # 손실함수 계산

        # 역전파
        optimizer.zero_grad() # 학습 수행 전 미분값을 0으로 초기화(학습전 반드시 처리 필요)
        loss.backward()       # 가중치와 편향에 대한 기울기 계산
        optimizer.step()      # 가중치와 편향 업데이트

        # 학습 정확도 및 손실함수 값 기록
        train_acc = get_accuracy(pred, y)  # 정확도 계산

        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc)

        if (batch+1) % num_batches == 0:
            print(f'step: {batch+1}/{num_batches} | {time.time() - start_time:.2f} s/step | ', end='')
            print(f'train loss: {np.mean(train_loss_list):.4f} | train acc: {np.mean(train_acc_list):.4f} | ', end='')

def valid(dataloader, model, criterion):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        val_loss_list, val_acc_list = [], []

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            correct = model(x)
            val_loss = criterion(correct, y)

            val_acc = get_accuracy(correct, y)

            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc)
        print(f'valid loss: {np.mean(val_loss_list):.4f} | valid acc: {np.mean(val_acc_list):.4f}')

def main():
    train_ds, valid_ds, classes = data_load()

    batch_size = 128
    learning_rate = 0.001
    num_epochs = 5
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = CNNClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        train(train_dl, model, criterion, optimizer)
        valid(valid_dl, model, criterion)
    print("Done!")   


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    main()