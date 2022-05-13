import network
import mnist_loader


nn = network.Network([2, 3, 2])         #  实例化对象

# 加载 .gz 数据集
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 定义3层神经网络，每层神经元个数分别为 784, 30, 10
net = network.Network([784, 30, 10])

# training_data = list(training_data)

# [((784 1),(10, 1)), ((784 1),(10, 1)), ...]
# for X, y in training_data:
#     print(X.shape, y.shape)
#     nabla_b, nabla_w = net.backprop(X, y)
#     break

# for b, w in zip(nabla_b, nabla_w):
#     print(nabla_b[0].shape, nabla_w[0].shape)
#     break

# [((784 1), 标签), ((784 1), 标签), ...]
# test_data = list(test_data)
# for X, y in test_data:
#     print(X.shape, y.shape)
#     print(y)
#     break


# for w, b in zip(net.weights, net.biases):
#     print(w.shape, b.shape)
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# for X, y in training_data:
#     a = net.feedforward(X)
#     print(a)
#     break

net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
