#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # 2D 합성곱층 정의
        conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0, dilation=1)
        conv2 = nn.Conv2d(6, 16, 5)
        # 풀링층 정의
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten layer 정의
        flatten = nn.Flatten()
        
        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(),
            pool,
            conv2, nn.ReLU(),
            pool, flatten
        )
        
        # 완전 연결층 정의(fully connected layer)
        fc1 = nn.Linear(5*5*16, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)
        # Dropout 층 정의
        dropout = nn.Dropout2d(0.5)

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(),
            fc2, nn.ReLU(),
            fc3, 
        )

    def forward(self, x):
        x = self.conv_module(x)
        #x = x.view(-1, 16 * 5 * 5)
        x = self.fc_module(x)
        output = F.softmax(x, dim=1)
        return output