import numpy as np
import pandas as pd
import random


class Network(object):
    '''
        sizes ： 各层神经元的个数
    '''
    def __init__(self, sizes :list):

        self.num_layers = len(sizes)                                  # 神经元的层数
        self.biases = [np.random.randn(y, 1)
                       for y in sizes[1:]]      # 偏置 b
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]  # 权重

    # 前馈传播
    def feedforward(self, a):
        """
            如果 a 是输入，则返回神经网络的输出。
            假如初始化为 3层的神经网络：[784, 30, 10]， 则：
            w = [(30, 784), (10, 30)]
            b = [(30, 1), (10, 1)]
            第一层时， a 的输入为 (784, 1), a 即是 激活层：通过激活函数得到的值，第一层时为样本输入。
            计算：
            第2层 : w · a + b :  (30, 784) · (784, 1) + (30, 1) ==> (30, 1)
            第3层 : w · a + b :  (10, 30) · (30, 1) + (10, 1)   ==> (10, 1)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)       # z = w · x + b   a = sigmoid(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ SGD : stochastic gradient descent
            training_data : 训练集，传入 zip() 对象， 使用 list() 后数据变为：[((784, 1), (10, 1)), ...]
            epochs : 训练轮数
            mini_batch_size : 批大小，每次训练的小样本批次大小
            eta : 学习率
            test_data : 可选参数， 对于追踪训练结果是有用的，但是会拖慢训练速度
        """
        training_data = list(training_data)       # 把 zip() 对象 ==> list [((784, 1), (10, 1)), ...]
        n = len(training_data)                    # 训练集样本个数

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)               # 测试集样本的长度

        for j in range(epochs):
            random.shuffle(training_data)         # 打乱(训练集)列表元素顺序
            # [[((784, 1), (10, 1)), ..., ], [mini_batch_size个],...]
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f"Epoch {j}:{self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} completed.")
        pass

    def update_mini_batch(self, mini_batch, eta):
        """使用梯度下降法和反向传播更新 mini_batch 的权重和偏置。
        mini_batch 类型如下： [((784, 1), (10, 1)), ...] 有 mini_batch_size 个样本(元组)
        eta : 学习率
        这两个算法比较复杂，梯度下降算法和反向传播算法的使用都是根据理论推导出的公式实现的，因此，我们只需要记住最后推导得出的公式进行实现即可。
        反向传步骤：
        对于每个样本 x：设置对应的激活值 a, 执行步骤如下：
            1.前向传播(feedforward): z = w · x + b , a = sigma(z)
            2.求输出误差: 从最后一层开始算起， 输出误差  delta = 偏C/偏a * sigma 的导数， 偏C/偏w = 前一层的a · 当前层的delta
            3.反向传播误差 ：从倒数第二层开始，到 正数第二场为止， 计算
                误差 倒数第二层的delta = w(最后后一层) · delta(最后一层的) * sigmoid的导数

        梯度下降步骤：对每个神经网络层， w_new = w_old - eta · delta · a^T
                                   b_new = b_old - eta * delta
        神经元成熟 和 每层神经元个数 ：[784, 30, 10]
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]    # [(30， 1), (10， 1)]
        nable_w = [np.zeros(w.shape) for w in self.weights]   # [(30, 784), [10, 30]]
        for x, y in mini_batch:
            # delta_nabla_b : (10, 1), delta_nable_w : (10, 30)
            delta_nabla_b, delta_nable_w = self.backprop(x, y)              # 反向传播由最后一层得到当前层的 偏置和权重的误差
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # b+delta b
            nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_w)] # w + delta w [(30, 784), (10, 30)]
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nable_w)] # w = w - eta/m * 偏C/偏w
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]   # b = b-eta/m * 偏C/偏b
        pass

    def backprop(self, x, y):
        """
            x : (784, 1) , ndarray 对象
            y : (10, 1) , ndarray 对象
            返回元组(nabla_b, nable_w) 代表成本/代价/损失函数 C_x 的 梯度(偏C/偏b, 偏C/偏w)。
            ([(30, 1), (10, 1)], [(30, 784), (10, 30)])

            前面注释可知公式：
            输出误差  delta = 偏C/偏a * sigma 的导数
            偏C/偏w = 误差delta · 前一层的激活层 a 的转秩
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # [(30， 1), (10， 1)]     存放偏置的误差 偏C/偏b
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # [(30, 784), [10, 30]]   存放权重的误差 偏C/偏w

        # 前向传播
        activation = x      # 激活值
        activations = [x]   # [(784, 1)] , list 列表 ： 存放所有的激活值,其中 a = sigmoid(z)

        zs = []  # list 存放每层的 z 向量（激活函数的输入值）， 其中 z = w · x + b [(30, 1), (10, 1)]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward 反向传播
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # 求最后一层的误差 delta = 偏C/偏a * sigma 的导数
        nabla_b[-1] = delta                          # 偏C/偏b
        # print(f"C/w:{delta.shape}, a前.shape:{activations[-2].shape}", )
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 偏C/偏w = 误差delta · 前一层的激活层 a 的转秩

        """
            前面已经求出了最后一层的 偏C/偏b 和 偏C/偏w
            从倒数第二层开始往前遍历到 正数第二层
        """
        for l in range(2, self.num_layers):
            z = zs[-l]              # 第一次遍历时，l = 2 ,表示倒数第二层 的激活函数的输入 z 向量
            sp = sigmoid_prime(z)   # 激活函数的倒数
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp     # delta = 后一层的w · 后一层的delta * 当前层的激活函数的倒数
            nabla_b[-l] = delta        # 误差、梯度、偏导数
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())     # 偏C/偏w = 误差delta · 前一层的激活层 a 的转秩
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations:np.ndarray, y:np.ndarray):
        """返回输出层层的误差。公式为
            误差 delta = 偏C_x / 偏a = 每层的激活值 - 真实值
            output_activations : (10, 1) - (10, 1) numpy.ndarray 类型
        """
        return (output_activations - y)


def sigmoid(z):
    """激活函数：sigmoid"""
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function
        Sigmoid 函数的偏导数
        直接查资料得到偏导数结果，推导可查看相关书籍
    """
    return sigmoid(z) * (1-sigmoid(z))