#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-07-29 (year-month-day)

"""
"""

import torch.nn as nn
import torch
from torch.optim import Adam, SGD


class MinNetwork(nn.Module):
    def __init__(self, n_units):
        super(MinNetwork, self).__init__()
        self.n_units = n_units
        self.par = nn.Parameter(torch.rand(1))
        self.dense = torch.eye(n_units*n_units, requires_grad=False) * self.par
        self.softmax = nn.LogSoftmax(dim=1)

    def get_penalty(self):
        foo = self.dense.weight

    def forward(self, mat):
        vec = mat.view(1, -1)
        out_vec = torch.matmul(vec, self.dense)
        out_mat = out_vec.view(self.n_units, self.n_units)
        return self.softmax(out_mat)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    def gen_data(size=3):
        x = torch.randn((size, size))
        y_vec = torch.argmax(x, 1)
        y = torch.zeros_like(x)
        for i, j in enumerate(y_vec):
            y[i, j] = 1.0
        return x, y_vec

    my_net = MinNetwork(3)

    my_optim = SGD(my_net.parameters(), lr=1e-3, momentum=0.9, weight_decay=100.)
    my_loss = nn.NLLLoss()

    buff_size = 100
    buff = torch.zeros(buff_size)
    x, y = gen_data()
    for i in range(10):
        my_optim.zero_grad()
        y_hat = my_net(x)
        print(x)
        print(y_hat)
        print(y)
        loss = my_loss(y_hat,y)
        buff[i % buff_size] = loss.item()
        # if i % 100 == 0:
        print(buff.mean())
        loss.backward()
        my_optim.step()
