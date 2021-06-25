#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
convolution layer의 출력층을 계산하기 위한 함수
'''
Hin = 32
Win = 32
kernel_size = (5, 5)


def param_calculatpr(Hin, Win, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):    
    Hout = ((Hin + (2 * padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    Wout = ((Win + (2 * padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
    return (Hout, Wout)