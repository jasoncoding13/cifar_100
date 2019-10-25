#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:32:23 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

from .lenet import LeNet


if __name__ == '__main__':
    lenet = LeNet(n_classes=3)
    lenet.build_graph()
    lenet.train()
