import numpy as np
import pandas as pd


class Network(object):
    '''
        sizes ： 各层神经元的个数
    '''
    def __init__(self, sizes :list):

        self.num_layers = len(sizes)                                  # 神经元的层数
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]      # 偏置 b
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    pass